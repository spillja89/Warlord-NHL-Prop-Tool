"""Official all-strength PP/PK windows, strictly before the saved slate date."""
from __future__ import annotations
import math
import pandas as pd

VERSION = 'pp-card-context-2026-11-07-v2'
FIELDS = {
    'Official_Team_OpportunitiesPG_L5': ('Team', 'L5', 'ppOpportunities', 'games', 1),
    'Official_Opp_FailurePct_L5': ('Opp', 'L5', 'ppGoalsAgainst', 'timesShorthanded', 100),
    'Official_Opp_G60_Season': ('Opp', 'Season', 'ppGoalsAgainst', 'pkTimeOnIcePerGame', 3600),
    'Official_Opp_G60_L5': ('Opp', 'L5', 'ppGoalsAgainst', 'pkTimeOnIcePerGame', 3600),
    'Official_Opp_SecondsPerOpportunity_L5': ('Opp', 'L5', 'pkTimeOnIcePerGame', 'timesShorthanded', 1),
    'Official_Opp_MinutesPG_L10': ('Opp', 'L10', 'pkTimeOnIcePerGame', 'games', 1/60),
}

def attach_pp_card_context(frame, before_day, reports, codes):
    """Never substitute season rates for rolling windows or unknown stats for zero."""
    out = frame.copy()
    if out.empty:
        return out
    before = str(before_day)[:10]
    if set(out.Date.astype(str).str[:10]) != {before}:
        raise ValueError('PPP cards require one exact pregame slate date')
    if not ('2026-07-01' <= before < '2027-07-01'):
        raise ValueError('PPP card kit requires the 2026-27 season')
    prepared = {}
    for side, report, cols in [('Team', 'powerplay', ['ppOpportunities']),
                                ('Opp', 'penaltykill', ['ppGoalsAgainst', 'timesShorthanded', 'pkTimeOnIcePerGame'])]:
        data = reports.get(report, pd.DataFrame()).copy()
        if data.empty:
            prepared[side] = data
            continue
        if not {'gameDate', 'teamId', 'gameId', *cols}.issubset(data.columns):
            raise ValueError('Incomplete official PPP report fields: ' + report)
        data = data.loc[data.gameDate.astype(str).str[:10].lt(before)].copy()
        if data.duplicated(['teamId', 'gameId']).any() or not data.gameId.astype(str).str.startswith('202602').all():
            raise ValueError('Invalid current-season regular-game identity')
        data['Team'] = data.teamId.map(codes)
        for col in cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data.loc[~data[col].map(lambda v: pd.notna(v) and math.isfinite(v) and v >= 0), col] = float('nan')
        prepared[side] = data.sort_values(['gameDate', 'gameId'])
    cache = {}
    def context(side, team):
        key = (side, team)
        if key in cache:
            return cache[key]
        source = prepared[side]
        history = source.loc[source.Team.eq(team)] if not source.empty else source
        values = {}
        for window, count in [('Season', None), ('L5', 5), ('L10', 10)]:
            recent = history if count is None else history.tail(count)
            prefix = f'PP_Card_{side}_{window}'
            values[prefix + '_GP'] = len(recent)
            values[prefix + '_Last_Date'] = str(recent.gameDate.max())[:10] if len(recent) else None
            for field, (which, period, numerator, denominator, multiplier) in FIELDS.items():
                if which != side or period != window:
                    continue
                value = None
                needed = [numerator] + ([] if denominator == 'games' else [denominator])
                if len(recent) and recent[needed].notna().all().all():
                    bottom = len(recent) if denominator == 'games' else float(recent[denominator].sum())
                    if bottom > 0:
                        value = multiplier * float(recent[numerator].sum()) / bottom
                values[field] = value
        cache[key] = values
        return values
    contexts = [{**context('Team', row.get('Team')), **context('Opp', row.get('Opp'))}
                for row in out.to_dict('records')]
    for field in sorted(set(FIELDS).union(*(v.keys() for v in contexts))):
        out[field] = [v.get(field) for v in contexts]
    out['PP_Card_Context_Version'] = VERSION
    out['PP_Card_Context_AsOf'] = before
    from pp_lower_context import attach_pp_lower_context
    out = attach_pp_lower_context(out, before, reports, codes)
    return out

