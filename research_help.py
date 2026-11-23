"""Plain-language display help. No data fetches, timers or model decisions."""
from html import escape
import re

CONF = 'The model’s player-profile score. Higher means a stronger score in that prop’s formula; 90 does not mean a 90% chance tonight.'
HELP = {
    'Green matrix': 'The player passes this prop’s Green screening rules. Green is a signal, not a promise of a hit.',
    'Conf_Assists': CONF, 'Conf_Points': CONF, 'Conf_Goal': CONF, 'Conf_SOG': CONF,
    'iXA%': 'How the player’s recorded assist total compares with other skaters in the model. A rank of 98 is higher than about 98% of that group. This is observed assists, not expected assists.',
    'iXG%': 'How the player’s individual expected-goal total ranks against other skaters in the model. It ranks shot quality and quantity, not the chance of scoring tonight.',
    'Assists_mu': 'The saved model’s estimated number of assists for this matchup. 1.10 means an average estimate of 1.10 assists, not a 110% chance.',
    'Points_mu': 'The saved model’s estimated number of points for this matchup. A point is a goal or an assist. This is an average estimate, not a hit percentage.',
    'Goal_mu': 'The saved model’s estimated number of goals for this matchup. 0.40 is an average estimate of 0.40 goals, not a 40% hit rate.',
    'Pregame_xG_L5_per_game': 'The player’s average individual expected goals over the last five completed games. More or better scoring chances raise xG. It is not actual goals or a forecast for tonight.',
    'TOI_per_game': 'Average minutes the player spends on the ice per game.',
    'TOI_Pct': 'Where the player’s average ice time ranks against other skaters. This is a ranking, not a share of the game.',
    'PP_TOI_per_game': 'Average minutes this player spends on the power play per game.',
    'EV5_TOI_per_game': 'Average minutes this player spends on the ice at five skaters against five.',
    'team_pp_share': 'The player’s average PP minutes divided by the team’s average PP minutes. 70% means they are on the ice for about seven of every ten team PP minutes.',
    'PP_TOI_min': 'Total power-play minutes in the source sample. This is a total, not minutes per game.',
    'PP_TOI_Pct': 'The player’s PP time as a percentage of their own total ice time. This differs from their share of the team’s PP time.',
    'PP_OnIce_xG_per_game': 'Expected goals created by the player’s team during their PP shifts, averaged per game. It includes teammates’ chances; it is not just this player’s shooting xG.',
    'PP_iXG_per_game': 'Expected goals from this player’s own shots on the power play, averaged per game.',
    'PP_iXA60': 'Recorded PP primary assists plus 0.75 times secondary assists, scaled to 60 PP minutes. The old iXA name is misleading: these are actual assists, not expected assists.',
    'PP_ActualPoints60': 'Actual goals plus assists on the power play, scaled to 60 PP minutes. It measures scoring involvement while on the PP.',
    'PP_Points60': 'A model proxy combining PP expected goals/60 with weighted PP assists/60. It is not actual points/60 or a predicted hit percentage.',
    'i5v5_primaryAssists60': 'Assists credited for the final pass before a goal at five-on-five, scaled to 60 minutes of five-on-five ice time.',
    'i5v5_SOG60': 'This player’s shots on goal at five-on-five, scaled to 60 minutes of five-on-five ice time.',
    'i5v5_xG60': 'Expected goals from this player’s own five-on-five shots, scaled to 60 minutes. It measures chance quality and quantity.',
    'PP_SOG60': 'Shots on goal taken by this player on the power play, scaled to 60 PP minutes.',
    'PPP10_total': 'Total credited PP points in the last ten completed games. A two-point game counts as two points here.',
    'Drought_G': 'Consecutive completed games without a goal. Zero means they scored in their last game. A drought does not guarantee a rebound.',
    'Drought_PPP': 'Consecutive completed games without a PP point. Zero means a PP point in the last game.',
    'ShotIntent': 'The model’s shooting-activity measure. It helps describe willingness to shoot; it is not a hit probability.',
    'ShotIntent_Pct': 'Where the player’s shot-intent score ranks against other skaters. 90 means higher than about 90% of that group.',
    'StarScore': 'The model’s overall player-quality score. It describes the profile; it is not a chance of hitting this prop.',
    'TalentMult': 'A multiplier the model uses for player quality. 1.20 means a 20% increase to the value it multiplies, not a 20-point hit-rate boost.',
    'Player_5v5_SOG_Share': 'The player’s share of the team’s five-on-five shots in this metric. A value of 10 means 10%, not ten shots.',
    'Player_SOG_Share_Proxy': 'A model proxy for the player’s share of team shooting. It describes offensive involvement, not a measured puck-possession share.',
    'Opp_SV': 'The opposing goalie’s save percentage: saves divided by shots faced. .910 means 91% saved. Higher is better for the goalie.',
    'Opp_GAA': 'The opposing goalie’s goals allowed per 60 minutes. 2.95 means 2.95 goals allowed per full game of ice time. Lower is better for the goalie.',
    'Goalie_GSAx_per_game': 'Expected goals faced minus actual goals allowed, per goalie appearance. Positive means the goalie allowed fewer than expected; negative means more than expected.',
    'Goalie_GSAA_per_game': 'Goals saved compared with an average save-percentage benchmark, per appearance. Positive is better for the goalie. This differs from shot-quality-based GSAx.',
    'opp_5v5_xGA60': 'Opponent expected goals allowed per 60 minutes at five-on-five. Higher means the opponent allows more or better scoring chances.',
    'opp_5v5_HDCA60': 'High-danger scoring chances allowed by the opponent per 60 minutes at five-on-five. Higher means more dangerous chances allowed.',
    'Opp_PK_xGA60': 'Expected goals allowed by the opponent per 60 minutes while killing penalties. Higher means more or better PP chances allowed.',
    'Opp_PK_GA60': 'Actual goals allowed by the opponent per 60 penalty-kill minutes. Higher means more PP goals conceded per minute of PK time.',
    'Opp_PK_pct': 'Percentage of opposing power plays the team kills without conceding a PP goal. Lower means a worse penalty-kill success rate.',
    'Team_PP_xGF60': 'Team expected goals created per 60 power-play minutes. Higher means more or better chances created on the PP.',
    'Team_PP_GF60': 'Actual team power-play goals per 60 PP minutes. This measures goals scored, not expected goals.',
    'NHL_Team_PPO_PG': 'Average power-play opportunities earned by the player’s team per game.',
    'NHL_Opp_PKO_PG': 'Average penalty-kill opportunities faced by the opponent per game.',
    'NHL_Opp_PK_Time_PG': 'Average minutes the opponent spends killing penalties per game. 4.5 minutes is 4 minutes 30 seconds.',
    'NHL_Opp_PK_Seconds_per_opportunity': 'Opponent’s total PK time divided by PK opportunities, in seconds. 90 means 1 minute 30 seconds per opportunity. PP goals can end a minor penalty early.',
    'NHL_Team_Rest_days': 'Full days off between the previous game and this game. Zero means back-to-back game days; one means one day off.',
    'PP unit': 'Projected power-play group: PP1 is the first unit. A complete projected lineup takes priority; season usage is the fallback. Units can change.',
    'Tonight PPP mean': 'Experimental estimate of how many PP points the player may record. This is an average estimate, not a hit probability.',
    'Tonight PPP 1+ %': 'Experimental model estimate of at least one PP point tonight. It is separate from the historical move hit rate.',
    'Player signals': 'The number of the four saved player checks passed after the PP on-ice xG entry. Expand the player signals to see the exact checks; 4/4 is not a probability.',
    'Team PP matchup signals': 'Opponent/team conditions found in the PP research. Teammates share these matchup flags. Each percentage belongs to its own historical team sample.',
    'BE': 'Book break-even: the hit rate needed to break even at those odds. +200 needs 33.3%; -150 needs 60%. Lower BE means a bigger payout per winning 1u bet.',
    'ROI': 'Profit divided by units staked. +20% ROI means 20 units profit for every 100 units staked in this record.',
    'Units': 'Profit or loss using the record’s stated staking basis. Move records use one unit risked per saved-price bet; these are not the owner’s actual stakes.',
    'Hit rate': 'Winning player-games divided by graded player-games in this move’s sample. It is a historical record, not tonight’s probability.',
    'S/A/B': 'Return grades use profit, ROI, sample size and recent health. They are separate from the Assist Crit labels and the player’s confidence score.',
    'Crit': 'For Assists: Crit I is 75–79.9% and Crit II is 80% or higher, with at least 30 historical plays. These labels describe the move record.',
}

HELP.update({
 'NHL_Team_Goals_per_shot_allowed_pct':'Player’s OWN team: goals conceded divided by opposing shots on goal, times100, before this slate. Higher means opponents converted more shots against this team. It is not the opponent’s save percentage or a causal scoring bonus.',
 'NHL_Opp_PK_L10_O_PG':'Opponent: average times shorthanded per game over its last ten completed games.',
 'NHL_Opp_PK_L5_Time_PG':'Opponent: average penalty-kill minutes per game over its last five completed games. Decimal minutes:5.15 means5:09.',
})

ALIASES = {
    'Conf':'Conf_Assists', 'Goal_Profile':'Conf_Goal',
    'GoalV2_Player_Log_L5_TOI_per_game':'Log_L5_TOI_per_game',
    'Player_Log_L5_TOI_per_game':'Log_L5_TOI_per_game',
    'GoalV2_i5v5_xG60':'i5v5_xG60', 'GoalV2_PP_SOG60':'PP_SOG60',
    'PP on-ice xG/game':'PP_OnIce_xG_per_game', 'PP on-ice xG/g':'PP_OnIce_xG_per_game',
    'PP shooter xG/game':'PP_iXG_per_game', 'Player PP iXG/g':'PP_iXG_per_game',
    'Opp PK %':'Opp_PK_pct', 'Opp PK xGA/60 now':'Opp_PK_xGA60',
    'Opp PK xGA/60':'Opp_PK_xGA60', 'Current PP min/game':'PP_TOI_per_game',
    'Team PP time share %':'team_pp_share', 'Team PP time %':'team_pp_share',
    'PP min/g':'PP_TOI_per_game', 'PP research signals':'Player signals',
    'PPP drought (games)':'Drought_PPP', 'PPP drought':'Drought_PPP',
    'NHL_Opp_PK_xGA60':'Opp_PK_xGA60', 'NHL_Opp_PK_G60':'Opp_PK_GA60',
    'NHL_Opp_PK_pct':'Opp_PK_pct', 'NHL_Team_PP_G60':'Team_PP_GF60',
}

HELP.update({
    'Assist_Dagger': 'A 0–100 blend of PP ice-time share, recorded PP assist production and PP matchup context. It describes the profile, not a chance of an assist.',
    'EV5_OZS_Pct': 'Offensive-zone starts divided by offensive- plus defensive-zone starts at five-on-five. It describes where shifts start, not time spent attacking.',
    'Player_L10_TOI_sd': 'How much ice time varies from game to game across up to ten completed games. Lower means steadier minutes; higher means more variation. 2.70 is 2 minutes 42 seconds of standard deviation, not the gap between the highest and lowest game.',
    'Player_L20_PPP_fraction_pct': 'Percentage of the player’s total points earned on the power play over up to twenty completed games. This is a share of points, not of ice time.',
    'Player_L20_PPP_per60': 'PP points per 60 minutes of TOTAL ice time over up to twenty completed games. The denominator includes non-PP minutes; this differs from points per 60 PP minutes.',
    'Player_L20_P_1plus_pct': 'Percentage of up to twenty completed games in which the player recorded at least one point.',
    'Player_Log_L20_G_hit_pct': 'Percentage of up to twenty completed games in which the player scored at least one goal.',
    'Player_Log_L20_S_hit_pct': 'Percentage of up to twenty completed games in which the player had at least three shots on goal, clearing a 2.5 line.',
    'Player_L20_P_total': 'Total goals plus assists in up to the last twenty completed games.',
    'Player_L20_G_total': 'Total goals in up to the last twenty completed games.',
    'L20_S': 'Total shots on goal in up to the last twenty completed games.',
    'Log_L20_S_median': 'The middle shot-on-goal total across up to twenty completed games. It shows a typical game with less influence from extremes.',
    'Player_L10_P_per60': 'Goals plus assists in up to the last ten games, scaled to 60 minutes of total player ice time.',
    'Player_L5_A_per60': 'Assists in up to the last five games, scaled to 60 minutes of total player ice time.',
    'Player_L5_P_2plus_pct': 'Percentage of up to the last five completed games with two or more points.',
    'Player_Pace_Gap_A10': 'Saved expected ten-game assist pace minus actual recent assist pace. Positive means actual assists ran below that saved baseline. It does not guarantee a rebound.',
    'Pace_Gap_P10': 'Saved expected ten-game points pace minus actual recent points pace. Positive means actual scoring ran below that baseline; negative means it ran above.',
    'Pace_Gap_G10': 'Saved expected ten-game goal pace minus actual recent goal pace. Positive means actual goals ran below that baseline. It does not establish that a player is due.',
    'SOG_Opportunity_Base': 'Five-on-five and PP shot rates multiplied by expected usage in each situation, then added. A descriptive shot-opportunity estimate; check source coverage before treating it as a full estimate.',
    'SOG_Opportunity_EV5': 'The player’s five-on-five shots per 60 minutes multiplied by average five-on-five minutes per game. It estimates shots from that usage and rate.',
    'PP_Matchup_Equal': 'An equal-weight PP context score from team offense and opponent penalty-kill conditions. It is a matchup summary, not a player’s PPP probability.',
    'PP_PPO_Blend_PG': 'Average of the team’s PP opportunities earned per game and the opponent’s PK opportunities faced per game. It is a matchup blend, not actual opportunities tonight.',
    'PP_Time_Blend_PG': 'Average of the team’s PP minutes per game and the opponent’s PK minutes per game. It is a matchup blend in minutes.',
    'Opp_5v5_xGA_relative_league': 'Opponent five-on-five expected goals allowed compared with the league rate. 1.05 means 5% above the league rate, not five percentage points.',
    'Opp_PK_xGA_relative_league': 'Opponent PK expected goals allowed compared with the league rate. 1.05 means 5% above the league rate.',
    'Team_PP_xGF_relative_league': 'Team PP expected goals created compared with the league rate. 1.10 means 10% above that rate.',
    'Opp_DefWeak': 'The model’s opponent defensive-weakness score. Higher means the model rates the defense as more vulnerable; it is not a scoring probability.',
    'team_5v5_SF60': 'Team shots on goal per 60 minutes of five-on-five play.',
    'Official_Opp_PK_L5_PPP_per_O': 'PP point credits conceded by the opponent divided by its PK opportunities over the last five games. One PP goal can credit a scorer and two assists, so this can exceed one.',
    'Official_Opp_PK_L10_Goals_PG': 'Average PP goals conceded by the opponent over its last ten completed games.',
    'Official_Opp_PK_L10_Seconds_per_O': 'Opponent PK seconds divided by PK opportunities over its last ten completed games. Shorter can reflect PP goals ending minor penalties early.',
    'Official_Opp_FailurePct_L5': 'PP goals conceded divided by PK opportunities over the opponent’s last five games, as a percentage. Higher means more PP goals per opportunity.',
    'Official_Opp_G60_L5': 'Actual PP goals conceded by the opponent per 60 PK minutes over its last five completed games.',
    'Official_Opp_G60_Season': 'Actual PP goals conceded by the opponent per 60 PK minutes this season.',
    'Official_Team_AssistCreditsPerGoal_L10': 'PP assists credited to the player’s team divided by its PP goals over the last ten games. A goal can have zero, one or two assists.',
    'MP_Team_PP_xG_minus_G60': 'Team five-on-four expected goals/60 minus actual goals/60. Positive means fewer goals scored than the chance-quality estimate; it does not guarantee future goals.',
    'Trend_Opp_PPPcreditsAllowedPG_L10': 'Opponent PP point credits allowed per game over the last ten games minus its season rate. Positive means more credits allowed recently. A goal can create up to three credits.',
})
ALIASES.update({
    'Current_Goal_Profile':'Conf_Goal', 'Current_SOG_Profile':'Conf_SOG',
    'Goalie_GAA':'Opp_GAA', 'GAA':'Opp_GAA', 'Goalie_SV':'Opp_SV',
    'Player_Rebuilt_Gap_G10':'Pace_Gap_G10',
    'Official_Opp_SecondsPerOpportunity_L10':'Official_Opp_PK_L10_Seconds_per_O',
    'Official_Player_MinutesPG_Season':'PP_TOI_per_game',
    'NHL_Team_PP_Season_G60':'Team_PP_GF60',
    'PP_TOI_Pct_Game':'PP_TOI_Pct',
})


_GOAL_V3_HELP = {'Player_Log_L20_TOI_median': 'Middle ice-time value across up to the last 20 completed games, in minutes.', 'Player_Log_L5_PPP_per60_total_TOI': 'Power-play points over the last five games per 60 minutes of TOTAL ice time. This is not per 60 PP minutes.', 'Player_Log_L5_S_sd': 'How much the player’s shots vary across the last five games. Lower means steadier shot totals.', 'Player_Log_L10_TOI_sd': 'How much ice time varies across up to the last ten games, in minutes. Lower means steadier usage.', 'Player_Log_L40_S_hit_pct': 'Percentage of available completed games, up to 40, with at least three shots. Uses the actual game count.', 'PP_TOI_Pct_Game': 'Percentage of this player’s TOTAL ice time spent on the power play. This is not team PP time share.', 'Goalie_Expected_SV': 'Expected share of shots saved based on the quality of shots faced: 1 minus xGA/shots. It is not actual save percentage or a goalie skill score.', 'Official_Opp_Season_GA_PG': 'Opponent TEAM goals allowed per game, all strengths. This is not the goalie’s GAA.', 'Official_Own_PP_Venue_O_PG': 'Own team’s earned PP opportunities per game in the relevant home or away games before this slate.', 'Official_Opp_PP_Season_G60': 'Opponent’s actual PP goals scored per 60 PP minutes before this slate.', 'Official_Opp_L10_SA_PG': 'Opponent’s average shots allowed in its last ten completed games.', 'Official_Opp_PK_L10_G60': 'Opponent’s goals allowed per 60 shorthanded minutes across its last ten completed games.', 'Official_Opp_PK_Season_O_PG': 'Opponent’s times shorthanded per game this season before the slate.'}

def field_help(field):
    if isinstance(field,str):
        _field=field.removeprefix('GoalV3_')
        if _field in _GOAL_V3_HELP:return _GOAL_V3_HELP[_field]
    # Display-only columns (for example PP matchup signals) have no raw field.
    # Let their caller fall back to the visible column label.
    if not isinstance(field,str) or not field.strip():
        return ''
    field = ALIASES.get(field, field)
    if field in HELP:
        return HELP[field]
    team_metric = re.fullmatch(r'NHL_(Team|Opp)_(GA|GF|SA|SF)_(5|10|L5|L10|Season|Venue)_PG', field)
    if team_metric:
        side, metric, window = team_metric.groups()
        who = 'Player’s team' if side == 'Team' else 'Opponent'
        stat = {'GA':'goals allowed', 'GF':'goals scored', 'SA':'shots on goal allowed', 'SF':'shots on goal taken'}[metric]
        period = 'at the relevant home/away venue' if window == 'Venue' else 'this season' if window == 'Season' else f'over its last {window.lstrip("L")} completed games'
        return f'{who}: average {stat} per game {period}.'
    special = re.fullmatch(r'NHL_(Team|Opp)_(PP|PK)_(Goals_PG|G60|Time_PG|Seconds_per_opportunity|L5_O_PG|L10_Time_PG|Season_SecondsPerOpportunity|L10_SecondsPerOpportunity|L5_Success_pct|Success_pct|Goal_drought|goal_drought)', field)
    if special:
        side, unit, metric = special.groups()
        who = 'Player’s team' if side == 'Team' else 'Opponent'
        situation = 'on the power play' if unit == 'PP' else 'while killing penalties'
        goals = 'goals scored' if unit == 'PP' else 'PP goals conceded'
        meaning = {'Goals_PG':f'average {goals} per game', 'G60':f'{goals} per 60 minutes {situation}', 'Time_PG':f'average minutes {situation} per game', 'Seconds_per_opportunity':f'seconds {situation} divided by opportunities', 'L5_O_PG':f'average opportunities {situation} per game over the last five games', 'L10_Time_PG':f'average minutes {situation} per game over the last ten games', 'Season_SecondsPerOpportunity':f'season seconds {situation} divided by opportunities', 'L10_SecondsPerOpportunity':f'seconds {situation} divided by opportunities over the last ten games', 'L5_Success_pct':'percentage of penalties killed without a PP goal over the last five games', 'Success_pct':('percentage of power plays producing a goal' if unit=='PP' else 'percentage of penalties killed without a PP goal'), 'Goal_drought':'consecutive completed games without conceding a PP goal', 'goal_drought':'consecutive completed games without conceding a PP goal'}[metric]
        return f'{who}: {meaning}.'
    if field in {'Official_Opp_MinutesPG_Season', 'Official_Opp_MinutesPG_L10', 'Official_Opp_SecondsPerOpportunity_L5', 'Official_Team_OpportunitiesPG_L5'}:
        return {'Official_Opp_MinutesPG_Season':'Average opponent penalty-kill minutes per game this season.', 'Official_Opp_MinutesPG_L10':'Average opponent penalty-kill minutes per game over its last ten games.', 'Official_Opp_SecondsPerOpportunity_L5':'Opponent PK seconds divided by PK opportunities over its last five games.', 'Official_Team_OpportunitiesPG_L5':'Average PP opportunities earned by the player’s team over its last five games.'}[field]
    if field in {'Avg5_SOG', 'L10_S', 'L10_G', 'L20_A', 'PPP L5', 'PPP L10', 'PPP season', 'Current PPP L5', 'Current PPP L10', 'Current PPP season'}:
        return {
            'Avg5_SOG':'Average shots on goal in the last five completed games.',
            'L10_S':'Total shots on goal in the last ten completed games.',
            'L10_G':'Total goals in the last ten completed games.',
            'L20_A':'Total assists in up to the last twenty completed games.',
        }.get(field, 'Games with at least one power-play point divided by games played in this window. A two-point game still counts as one hit. The window uses completed games before this slate.')
    if field in {'Trim10_SOG', 'Opp_SOG_Against_L10'}:
        return {'Trim10_SOG':'The last-ten-game shots average after trimming the extreme values. This reduces the effect of unusually high or low games.', 'Opp_SOG_Against_L10':'Average shots on goal the opponent allowed over its last ten completed games.'}[field]
    pattern = re.fullmatch(r'(?:Player_)?Log_L(5|10|20)_(TOI_per_game|Shooting_pct|PPP_1plus_pct|PPP_hit_pct|P_2plus_pct|Point2plus_pct|P_per_game|S_per_game|A_per_game|Goal_fraction_pct)', field)
    if pattern:
        games, kind = pattern.groups()
        meaning = {'TOI_per_game':'Average actual ice time in minutes per game', 'Shooting_pct':'Goals divided by shots on goal, as a percentage', 'PPP_1plus_pct':'Percentage of games with at least one PP point', 'PPP_hit_pct':'Percentage of games with at least one PP point', 'P_2plus_pct':'Percentage of games with two or more points', 'Point2plus_pct':'Percentage of games with two or more points', 'P_per_game':'Average points per game', 'S_per_game':'Average shots on goal per game', 'A_per_game':'Average assists per game', 'Goal_fraction_pct':'Goals as a percentage of the player’s goals plus assists'}[kind]
        return f'{meaning} over the last {games} completed games available before this slate. Missing games are not counted as zero.'
    return ''


def hint(label, field=None, *, text=None):
    explanation = text or field_help(field or label)
    if not explanation:
        return escape(str(label))
    return ('<abbr class="wl-help" tabindex="0" title="' + escape(explanation, quote=True)
            + '" aria-label="' + escape(str(label) + '. ' + explanation, quote=True)
            + '" style="cursor:help;text-decoration:underline dotted;text-underline-offset:3px">'
            + escape(str(label)) + '</abbr>')


def condition_hint(part, labels=None):
    if part == 'Green matrix':
        return hint(part)
    for field, label in sorted((labels or {}).items(), key=lambda item: len(item[1]), reverse=True):
        if part.startswith(label + ' '):
            return hint(part, field)
    raw = re.match(r'^([\w%]+)\s*(?:>=|<=|>|<|≥|≤)', part)
    if raw:
        return hint(part, raw[1])
    return escape(part)


CHAMPIONS = {
    'Arcanist': 'Gorilla · Arcanist: the core playmaker. The leading move follows the broad assist profile or a facilitator route.',
    'Whisperer': 'Gorilla · Whisperer: an assist route with a ceiling on recent shots. It describes a less shot-heavy profile; it does not mean avoiding shots causes assists.',
    'Oracle': 'Gorilla · Oracle: the facilitator. Routes look for assist-oriented involvement, including lower saved goal projection or primary-assist patterns, depending on the exact move.',
    'Runesmith': 'Orangutan · Runesmith: PP production and involvement routes. Different moves use actual PP assists, PP points or recent PP production; the move combo shows which.',
    'Oracle + Runesmith': 'Orangutan · Oracle + Runesmith: the facilitator route and PP-production route both qualify, with the named environment conditions. This is the Runic Orchestration combination.',
    'Stormcaller': 'Chimp · Stormcaller: offensive involvement routes. Depending on the move, it uses shot share or heavier PP ice time. Check the combo for the exact requirements.',
}


CLASS_CHAMPIONS = {
    'Carry': {
        'Ravager': 'Gorilla · Ravager: Goal routes using a less assist-heavy recent profile or the saved goal-pace gap. The exact move decides which checks apply.',
        'Reaper': 'Chimp · Reaper: Goal routes using recent shooting conversion or five-on-five chance-production ranges. The leading move shows the exact route.',
        'Executioner': 'Panda · Executioner: Goal routes using PP shooting activity or a larger goal share of the player’s points. These are separate paths, not universal requirements.',
    },
    'Tank': {
        'Bulwark': 'Gorilla · Bulwark: the look used for the broader 0.5-point attack routes. A point can come from a goal or an assist.',
        'Siegebreaker': 'Orangutan · Siegebreaker: the look used for Heavy, Special or Crit-style 0.5-point routes. The move combo explains the added requirements.',
        'War Drummer': 'Panda · War Drummer: the 1.5-point specialist. The player needs two or more points, from any combination of goals and assists.',
    },
    'Jungle': {
        'Berserker': 'Gorilla · Berserker: the core shot-profile routes, including saved shooting history, scoring involvement or drought conditions depending on the move.',
        'Ranger': 'Orangutan · Ranger: routes using recent minutes, shot history or a rate-times-usage shot-opportunity measure. They are tested profile ranges, not simply the most ice time.',
        'Warden': 'Panda · Warden: shot routes with a tested matchup condition, such as the opponent’s recent penalty-kill goal history. Inspect shows the exact condition.',
    },
}


def champion_hint(label, variant, role='Support'):
    descriptions = CHAMPIONS if role == 'Support' else CLASS_CHAMPIONS.get(role, {})
    return hint(label, text=descriptions.get(variant))


def support_guide():
    rows = [('<b>Gorilla · Arcanist / Oracle</b>', 'Playmaker and facilitator routes.'),
            ('<b>Orangutan · Runesmith</b>', 'PP production routes; Oracle + Runesmith combines facilitation with PP involvement.'),
            ('<b>Chimp · Stormcaller</b>', 'Offensive involvement through shot-share or PP-usage routes.')]
    body = ''.join('<p style="margin:7px 0">' + label + ' — ' + text + '</p>' for label, text in rows)
    terms = ' · '.join(hint(label, field) for label, field in [('Confidence', 'Conf_Assists'), ('Assist rank', 'iXA%'), ('Assist μ', 'Assists_mu'), ('Book BE', 'BE'), ('Units', 'Units'), ('ROI', 'ROI'), ('S/A/B tiers', 'S/A/B'), ('Crit I / II', 'Crit')])
    return ('<details class="wn-champion-guide" style="margin:10px 14px;padding:10px 12px;border:1px solid #74658c;border-radius:7px;color:#e6e1f1;font-size:12px">'
            '<summary style="cursor:pointer;font-weight:800;color:#e5cbff">Meet the Assist champions · how to read the cards</summary>'
            + body + '<p>The leading move chooses the portrait. A player may match more than one route; the character is not their tier. Inspect shows the other routes.</p>'
            '<p>Hover dotted labels for a plain-English explanation. On a phone, this guide explains the main terms:</p>'
            + '<p>' + terms + '</p>'
            '<p><b>Confidence</b> is a profile score. <b>Assist rank</b> compares recorded assist totals. <b>μ</b> is an average model estimate. <b>BE</b> is the win rate needed at the odds. '
            '<b>Units</b> are profit at the record’s stated stake; <b>ROI</b> is profit divided by stakes. <b>S/A/B</b> grade returns; <b>Crit I/II</b> describe historical Assist hit rates.</p></details>')


def class_guide(role):
    if role == 'Support':
        return support_guide()
    if role not in CLASS_CHAMPIONS:
        return ''
    market, purpose = {
        'Carry': ('Goal', 'Carry hunts goals. Over 0.5 means the player must score at least once.'),
        'Tank': ('Points', 'Tank looks for scoring involvement. One goal or one assist is one point; over 1.5 needs at least two points.'),
        'Jungle': ('Shot', 'Jungle looks for shots on goal. Over 2.5 needs at least three; attempts that miss the net or are blocked are not shots on goal.'),
    }[role]
    body = ''.join('<p style="margin:7px 0">' + hint(description.split(': ', 1)[0], text=description) + ' — '
                   + escape(description.split(': ', 1)[1]) + '</p>'
                   for variant, description in CLASS_CHAMPIONS[role].items())
    return ('<details class="wn-champion-guide" style="margin:10px 14px;padding:10px 12px;border:1px solid #74658c;border-radius:7px;color:#e6e1f1;font-size:12px">'
            '<summary style="cursor:pointer;font-weight:800;color:#e5cbff">Meet the ' + market
            + ' champions · how to read the cards</summary><p>' + purpose + '</p>' + body
            + '<p>The leading move chooses the portrait. A player can match several routes; the animal is not their return tier. Inspect shows the others.</p>'
            '<p><b>Confidence</b> is a player-profile score. <b>Move %</b> is a historical hit rate. <b>BE</b> is the hit rate needed at the posted odds. '
            '<b>Units</b> show record profit; <b>ROI</b> is profit divided by stakes. <b>S/A/B</b> grade returns and recent health. '
            'Buffs add researched matchup context; they are not extra independent bets. Hover dotted stat labels for details.</p></details>')
