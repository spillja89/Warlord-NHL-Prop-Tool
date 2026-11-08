"""Exact pregame sources for the approved lower-on-ice-xG PPP lanes."""
from __future__ import annotations
import math
import pandas as pd

VERSION = 'pp-lower-context-2026-11-07-v1'
FIELDS = ('Official_Opp_MinutesPG_Season', 'Official_Opp_SecondsPerOpportunity_L10',
          'Official_Player_MinutesPG_Season', 'Official_Team_AssistCreditsPerGoal_L10',
          'Trend_Opp_PPPcreditsAllowedPG_L10', 'MP_Team_PP_xG_minus_G60')

def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def _prepare(source, before, identity, required):
    if source.empty or not {'gameDate', 'gameId', *identity, *required}.issubset(source.columns):
        return pd.DataFrame()
    data = source.loc[source.gameDate.astype(str).str[:10].lt(before)].copy()
    if data.duplicated([*identity, 'gameId']).any() or not data.gameId.astype(str).str.startswith('202602').all():
        raise ValueError('Invalid official lower-PPP game identity')
    for col in required:
        if col in ('teamAbbrev', 'opponentTeamAbbrev'):
            continue
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data.loc[~data[col].map(lambda v: pd.notna(v) and math.isfinite(v) and v >= 0), col] = float('nan')
    return data.sort_values(['gameDate', 'gameId'])

def attach_pp_lower_context(frame, before_day, reports, codes):
    out = frame.copy()
    if out.empty:
        return out
    before = str(before_day)[:10]
    if set(out.Date.astype(str).str[:10]) != {before} or not ('2026-07-01' <= before < '2027-07-01'):
        raise ValueError('Lower-PPP context requires one 2026-27 pregame date')
    pp = _prepare(reports.get('powerplay', pd.DataFrame()), before, ['teamId'],
                  ['powerPlayGoalsFor', 'opponentTeamAbbrev'])
    pk = _prepare(reports.get('penaltykill', pd.DataFrame()), before, ['teamId'],
                  ['pkTimeOnIcePerGame', 'timesShorthanded', 'ppGoalsAgainst'])
    sk = _prepare(reports.get('skater_powerplay', pd.DataFrame()), before, ['playerId'],
                  ['ppTimeOnIce', 'ppGoals', 'ppAssists', 'ppPoints', 'teamAbbrev'])
    for source in (pp, pk):
        if not source.empty:
            source['Team'] = source.teamId.map(codes)
    # Keep a missing/incomplete game missing; never turn absent credits into zero.
    credits = {}
    if not sk.empty:
        for (game, team), group in sk.groupby(['gameId', 'teamAbbrev']):
            cols = group[['ppGoals', 'ppAssists', 'ppPoints']]
            if cols.notna().all().all() and group.ppPoints.eq(group.ppGoals + group.ppAssists).all():
                g, a, p = (float(cols[c].sum()) for c in cols)
                if a <= 2*g:
                    credits[(int(game), str(team))] = (g, a, p)
    pp_groups = {} if pp.empty else {t: h for t, h in pp.groupby('Team')}
    pk_groups = {} if pk.empty else {t: h for t, h in pk.groupby('Team')}
    player_groups = {} if sk.empty else {int(p): h for p, h in sk.groupby('playerId')}
    team_games = {} if pp.empty else {(int(r.gameId), r.Team): r for r in pp.itertuples()}
    cache = {}
    def team_values(team, opp):
        key = (team, opp)
        if key in cache:
            return cache[key]
        z = {}
        own, defense = pp_groups.get(team, pd.DataFrame()), pk_groups.get(opp, pd.DataFrame())
        z['PP_Lower_Team_L10_GP'] = min(10, len(own))
        z['PP_Lower_Opp_Season_GP'] = len(defense)
        z['PP_Lower_Opp_L10_GP'] = min(10, len(defense))
        for label, history in [('Team', own), ('Opp', defense)]:
            z['PP_Lower_'+label+'_Last_Date'] = str(history.gameDate.max())[:10] if len(history) else None
        if len(defense):
            if defense.pkTimeOnIcePerGame.notna().all():
                z['Official_Opp_MinutesPG_Season'] = float(defense.pkTimeOnIcePerGame.sum())/60/len(defense)
            recent = defense.tail(10)
            if recent[['pkTimeOnIcePerGame', 'timesShorthanded']].notna().all().all() and recent.timesShorthanded.sum() > 0:
                z['Official_Opp_SecondsPerOpportunity_L10'] = float(recent.pkTimeOnIcePerGame.sum()/recent.timesShorthanded.sum())
            points = []
            for row in defense.itertuples():
                game = team_games.get((int(row.gameId), opp))
                credit = credits.get((int(row.gameId), str(game.opponentTeamAbbrev))) if game is not None else None
                points.append(credit[2] if credit is not None and credit[0] == row.ppGoalsAgainst else None)
            if all(v is not None for v in points):
                z['Trend_Opp_PPPcreditsAllowedPG_L10'] = sum(points[-10:])/min(10,len(points))-sum(points)/len(points)
        if len(own):
            recent = own.tail(10)
            own_credits = [credits.get((int(row.gameId), team)) for row in recent.itertuples()]
            if all(c is not None and c[0] == goal for c, goal in zip(own_credits, recent.powerPlayGoalsFor)):
                goals = sum(c[0] for c in own_credits)
                if goals > 0:
                    z['Official_Team_AssistCreditsPerGoal_L10'] = sum(c[1] for c in own_credits)/goals
        cache[key] = z
        return z
    values = []
    for row in out.to_dict('records'):
        z = dict(team_values(row.get('Team'), row.get('Opp')))
        pid = _num(row.get('Player_ID'))
        history = player_groups.get(int(pid), pd.DataFrame()) if pid is not None and pid.is_integer() else pd.DataFrame()
        z['PP_Lower_Player_Season_GP'] = len(history)
        z['PP_Lower_Player_Last_Date'] = str(history.gameDate.max())[:10] if len(history) else None
        if len(history) and history.ppTimeOnIce.notna().all():
            z['Official_Player_MinutesPG_Season'] = float(history.ppTimeOnIce.sum())/60/len(history)
        expected, actual = _num(row.get('Team_PP_xGF60')), _num(row.get('Team_PP_GF60'))
        if row.get('Team_Stats_Season') == '2026-2027' and expected is not None and actual is not None and min(expected,actual) >= 0:
            z['MP_Team_PP_xG_minus_G60'] = expected-actual
        values.append(z)
    for field in sorted(set(FIELDS).union(*(z.keys() for z in values))):
        out[field] = [z.get(field) for z in values]
    out['PP_Lower_Context_Version'] = VERSION
    out['PP_Lower_Context_AsOf'] = before
    return out
