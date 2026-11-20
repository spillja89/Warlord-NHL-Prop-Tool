"""Reuse validated public inputs only inside one Run / Refresh invocation."""
from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit
import requests


def ordered_map(function, items, workers=4):
    """Bound independent public lookups; retain input order and every result."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(function, items))


class RunResources:
    """A fresh wrapper per model run, never a cache shared with a later run.

    Odds, starter/injury reports, errors, partial downloads and unfinished
    gamecenter responses are never reused. All original source calls and
    parsers remain responsible for validating coverage and units.
    """
    def __init__(self, session):
        self.session = session
        self._lock = Lock()
        self._pending = {}
        self._frames = {}
        self.network_calls = self.reused_calls = self.csv_reuses = 0
        self.network_seconds = 0.0
        self._started = self._mark = perf_counter()

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _key(self, url, kwargs):
        if kwargs.get('stream') or kwargs.get('auth') or kwargs.get('cookies'):
            return None
        if getattr(self.session, 'auth', None):
            return None
        if any(str(k).lower() in {'authorization', 'cookie'} for k in kwargs.get('headers', {})):
            return None
        if any(str(k).lower() in {'authorization', 'cookie'} for k in getattr(self.session, 'headers', {})):
            return None
        prepared = requests.Request('GET', url, params=kwargs.get('params')).prepare().url
        parts = urlsplit(prepared)
        query = parse_qsl(parts.query, keep_blank_values=True)
        if any(k.lower() in {'apikey', 'api_key', 'token', 'key'} for k, _ in query):
            return None
        allowed = (parts.hostname == 'api.nhle.com' and parts.path.startswith('/stats/rest/en/'))
        if parts.hostname == 'api-web.nhle.com':
            allowed = any(parts.path.startswith('/v1/' + prefix) for prefix in (
                'club-schedule-season/', 'player/', 'roster/', 'gamecenter/'))
        if not allowed:
            return None
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sorted(query)), ''))

    @staticmethod
    def _clone(response):
        out = requests.Response()
        out.status_code = response.status_code
        out._content = bytes(response.content)
        out._content_consumed = True
        out.headers = response.headers.copy()
        out.encoding, out.url, out.reason = response.encoding, response.url, response.reason
        out.request = response.request
        return out

    def get(self, url, **kwargs):
        key = self._key(url, kwargs)
        future, owner = None, True
        if key is not None:
            with self._lock:
                future = self._pending.get(key)
                owner = future is None
                if owner:
                    future = Future()
                    self._pending[key] = future
                else:
                    self.reused_calls += 1
            if not owner:
                saved = future.result()
                return self._clone(saved) if saved is not None else self.get(url, **kwargs)
        start = perf_counter()
        try:
            with self._lock:
                self.network_calls += 1
            response = self.session.get(url, **kwargs)
            reusable = response.status_code == 200 and not kwargs.get('stream')
            if future is not None:
                try:
                    payload = response.json() if reusable else None
                    reusable = reusable and isinstance(payload, (dict, list))
                    if '/gamecenter/' in url:
                        reusable = reusable and isinstance(payload, dict) and payload.get('gameState') in {'OFF', 'FINAL'}
                except (ValueError, requests.RequestException):
                    reusable = False
                if not reusable:
                    with self._lock:
                        self._pending.pop(key, None)
                future.set_result(self._clone(response) if reusable else None)
            return response
        except BaseException as error:
            if future is not None:
                future.set_exception(error)
                with self._lock:
                    self._pending.pop(key, None)
            raise
        finally:
            with self._lock:
                self.network_seconds += perf_counter() - start

    def frame(self, url):
        with self._lock:
            data = self._frames.get(url)
            if data is None:
                return None
            self.csv_reuses += 1
            return data.copy(deep=True)

    def remember_frame(self, url, data):
        # Called only after the original complete CSV parser succeeds.
        with self._lock:
            self._frames[url] = data.copy(deep=True)

    def mark(self, phase):
        now = perf_counter()
        print(f'[Run timing] {phase}: {now-self._mark:.2f}s; elapsed {now-self._started:.2f}s', flush=True)
        self._mark = now

    def report(self):
        print(f'[Run downloads] {self.network_calls} requests; {self.reused_calls} duplicate public JSON downloads avoided; '
              f'{self.csv_reuses} validated CSV reuses; elapsed {perf_counter()-self._started:.2f}s', flush=True)
