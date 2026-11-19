"""Complete NHL Stats reports, including servers that cap each page at 100."""
from __future__ import annotations


def fetch_all_rows(session, url, *, params=None, timeout=20):
    query = dict(params or {})
    query.update(start=0, limit=100)
    rows, expected, signatures = [], None, set()
    while True:
        response = session.get(url, params=query.copy(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        page = payload.get('data')
        if not isinstance(page, list):
            raise ValueError('Invalid NHL Stats report data')
        total = int(payload.get('total', len(page)))
        if total >= 10000:
            raise ValueError('NHL Stats report needs a narrower date window')
        if expected is None:
            expected = total
        if total != expected or total < 0:
            raise ValueError('NHL Stats report changed during pagination')
        import json
        signature = json.dumps(page, sort_keys=True, separators=(',', ':'))
        if page and signature in signatures:
            raise ValueError('NHL Stats repeated a report page')
        signatures.add(signature)
        rows.extend(page)
        if len(rows) == expected:
            return rows
        if not page or len(rows) > expected:
            raise ValueError('Incomplete NHL Stats report')
        query['start'] = len(rows)
