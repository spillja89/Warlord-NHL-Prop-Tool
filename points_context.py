"""Strictly pre-slate official team context for the approved Points profiles."""
from __future__ import annotations
import json
import math
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

VERSION = 'points-context-2026-11-05-v1'

def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def log_context(row):
    """Actual up-to20-game denominator; incomplete PPP history stays missing."""
    result = {}
    try:
        data = json.loads(row['Form_Log']) if isinstance(row.get('Form_Log'), str) else row['Form_Log']
        if not isinstance(data, dict) or data.get('season') != '2026-27':
            return result
        before = str(row.get('Date', ''))[:10]
        games = sorted([g for g in data['games'] if str(g.get('d', '')) < before], key=lambda g: g['d'], reverse=True)[:20]
        if not games:
            return result
        points = [_num(g.get('g')) + _num(g.get('a')) for g in games]
        result['Log_L20_Point2plus_pct'] = 100 * sum(a >= 2 for a in points) / len(points)
        pp = [_num(g.get('pp')) for g in games]
        if all(a is not None for a in pp):
            result['Log_L20_PPP_hit_pct'] = 100 * sum(a > 0 for a in pp) / len(pp)
    except (KeyError, TypeError, ValueError):
        pass
    return result

def fetch_reports(session, season):
    base = 'https://api.nhle.com/stats/rest/en/team'
    lookup = session.get(base, timeout=20)
    lookup.raise_for_status()
    codes = {int(z['id']): str(z['triCode']).upper() for z in lookup.json().get('data', []) if z.get('triCode')}
    params = {'cayenneExp': f'seasonId={season} and gameTypeId=2', 'isAggregate': 'false', 'isGame': 'true', 'limit': -1, 'start': 0}
    def fetch(name):
        response = session.get(base + '/' + name, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get('data', [])
        if int(payload.get('total', len(rows))) != len(rows):
            raise ValueError('Incomplete official team-game report: ' + name)
        return name, pd.DataFrame(rows)
    with ThreadPoolExecutor(max_workers=3) as pool:
        reports = dict(pool.map(fetch, ['summary', 'powerplay', 'penaltykill']))
    return reports, codes

def attach_points_context(frame, session=None, before_day=None, reports=None, codes=None):
    out = frame.copy()
    if out.empty:
        return out
    dates = out['Date'].astype(str).str[:10].unique()
    before = str(before_day or dates[0])[:10]
    if len(dates) != 1 or dates[0] != before:
        raise ValueError('Points context requires one exact slate date')
    values = [log_context(row) for row in out.to_dict('records')]
    for field in ['Log_L20_Point2plus_pct', 'Log_L20_PPP_hit_pct']:
        out[field] = [z.get(field) for z in values]
    if reports is None:
        try:
            if session is None:
                import requests
                session = requests.Session()
            reports, codes = fetch_reports(session, 20262027)
        except Exception as error:
            print('[Points context] Official history unavailable: ' + type(error).__name__)
            reports, codes = {}, {}
    prepared = {}
    for name, source in reports.items():
        data = source.copy()
        if data.empty:
            continue
        data = data.loc[data.gameDate.astype(str).str[:10].lt(before)].copy()
        if data.duplicated(['teamId', 'gameId']).any() or not data.gameId.astype(str).str.startswith('202602').all():
            raise ValueError('Invalid official current-season team-game identity')
        data['Team'] = data.teamId.map(codes)
        prepared[name] = data.sort_values(['gameDate', 'gameId'])
    cache = {}
    def team_context(team, site):
        key = (team, site)
        if key in cache:
            return cache[key]
        z = {}
        summary = prepared.get('summary', pd.DataFrame())
        if summary.empty:
            cache[key] = z
            return z
        summary = summary.loc[summary.Team.eq(team)]
        if summary.empty:
            cache[key] = z
            return z
        gp = len(summary)
        z.update(Prior_GP=gp, Latest_source_date=str(summary.gameDate.max())[:10])
        for window in ['Season', 5, 10]:
            recent = summary if window == 'Season' else summary.tail(window)
            for field, source in [('GF', 'goalsFor'), ('GA', 'goalsAgainst'), ('SF', 'shotsForPerGame'), ('SA', 'shotsAgainstPerGame')]:
                z[f'{field}_{window}_PG'] = float(recent[source].mean())
        venue = summary.loc[summary.homeRoad.eq(site)]
        z['Venue_GP'] = len(venue)
        for field, source in [('GF', 'goalsFor'), ('GA', 'goalsAgainst'), ('SF', 'shotsForPerGame'), ('SA', 'shotsAgainstPerGame')]:
            z[f'{field}_Venue_PG'] = float(venue[source].mean()) if len(venue) >= 3 else None
        z['Goal_conversion_pct'] = 100 * summary.goalsFor.sum() / summary.shotsForPerGame.sum()
        z['Goals_per_shot_allowed_pct'] = 100 * summary.goalsAgainst.sum() / summary.shotsAgainstPerGame.sum()
        z['Win_pct'] = 100 * summary.wins.sum() / gp
        z['Point_pct'] = 100 * summary.points.sum() / (2 * gp)
        for side, name, opcol, timecol, goalcol in [('PP', 'powerplay', 'ppOpportunities', 'ppTimeOnIcePerGame', 'powerPlayGoalsFor'), ('PK', 'penaltykill', 'timesShorthanded', 'pkTimeOnIcePerGame', 'ppGoalsAgainst')]:
            report = prepared.get(name, pd.DataFrame())
            if report.empty:
                continue
            report = report.loc[report.Team.eq(team)]
            if set(report.gameId) != set(summary.gameId):
                continue
            ops, seconds, goals = float(report[opcol].sum()), float(report[timecol].sum()), float(report[goalcol].sum())
            if ops <= 0 or seconds <= 0:
                continue
            z.update({side + 'O_PG': ops / gp, side + '_Time_PG': seconds / gp / 60,
                      side + '_Goals_PG': goals / gp, side + '_G60': 3600 * goals / seconds,
                      side + '_Seconds_per_opportunity': seconds / ops,
                      side + '_Success_pct': 100 * (goals / ops if side == 'PP' else 1 - goals / ops),
                      side + '_goal_drought': next((i for i, a in enumerate(report[goalcol].iloc[::-1]) if a > 0), len(report))})
        cache[key] = z
        return z
    contexts = []
    for row in out.to_dict('records'):
        game = str(row.get('Game', '')).replace(' ', '').split('@')
        site = 'H' if len(game) == 2 and str(row.get('Team')) == game[1] else 'R'
        z = {}
        for side, team, venue in [('Team', row.get('Team'), site), ('Opp', row.get('Opp'), 'R' if site == 'H' else 'H')]:
            z.update({'NHL_' + side + '_' + field: value for field, value in team_context(team, venue).items()})
        for target, numerator, denominator in [('Team_PP_xGF_relative_league', 'Team_PP_xGF60', 'League_PP_xGF60'), ('Opp_PK_xGA_relative_league', 'Opp_PK_xGA60', 'League_PK_xGA60'), ('Opp_5v5_xGA_relative_league', 'opp_5v5_xGA60', 'League_5v5_xGA60')]:
            a, b = _num(row.get(numerator)), _num(row.get(denominator))
            z[target] = a / b if row.get('Team_Stats_Season') == '2026-2027' and a is not None and b is not None and b > 0 else None
        contexts.append(z)
    fields = set().union(*(z.keys() for z in contexts))
    for field in sorted(fields):
        out[field] = [z.get(field) for z in contexts]
    out['Points_Context_Version'] = VERSION
    # Reuse these exact official reports; no additional download during a run.
    from pp_card_context import attach_pp_card_context
    out = attach_pp_card_context(out, before, reports, codes or {})
    return out

