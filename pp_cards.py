"""Approved PPP profiles; frozen records are distinct from forecasts."""
from __future__ import annotations
from functools import lru_cache
from html import escape
from pathlib import Path
import json
import math

from pp_card_context import VERSION as CONTEXT_VERSION
from pp_lower_context import VERSION as LOWER_CONTEXT_VERSION
from special_return_monitor import pp_return_panel
from goal_return_monitor import RETURN_CSS
from move_return_grades import pp_move_rank, move_grade, grade_badges, GRADE_CSS
from pp_card_form import ppp_l10_chart, ppp_l10_label, FORM_CSS

VERSION = 'pp-card-kit-2026-11-07-v2'
LABELS = {
    'Official_Opp_FailurePct_L5': ('Opponent L5 PK goals / opportunities', '%'),
    'Official_Team_OpportunitiesPG_L5': ('Team L5 PP opportunities/game', ''),
    'Official_Opp_G60_Season': ('Opponent season PK GA/60', ''),
    'Official_Opp_G60_L5': ('Opponent L5 PK GA/60', ''),
    'Official_Opp_SecondsPerOpportunity_L5': ('Opponent L5 PK seconds/opportunity', ' sec'),
    'Official_Opp_MinutesPG_L10': ('Opponent L10 PK minutes/game', ' min'),
    'Official_Opp_MinutesPG_Season': ('Layer 1 · opponent season PK minutes/game', ' min'),
    'Official_Opp_SecondsPerOpportunity_L10': ('Opponent L10 PK seconds/opportunity', ' sec'),
    'Official_Player_MinutesPG_Season': ('Official player PP minutes/game', ' min'),
    'Official_Team_AssistCreditsPerGoal_L10': ('Team L10 PP assists credited/goal', ''),
    'MP_Team_PP_xG_minus_G60': ('Team 5on4 xGF/60 minus actual GF/60', ''),
    'Trend_Opp_PPPcreditsAllowedPG_L10': ('Opponent PPP allowed/game · L10 minus season', ''),
    'Assists_mu': ('Saved Assist μ', ''),
    'iXA%': ('Observed overall assist percentile', ''),
    'PP_OnIce_xG_per_game': ('PP on-ice xG/game', ''),
}

RUNESMITH_IDS = {'pp-arcane-momentum-i-v1', 'pp-arcane-momentum-ii-v1',
                 'pp-rune-rupture-v1', 'pp-arcane-overload-v1',
                 'pp-runic-weave-i-v1', 'pp-runic-weave-ii-v1',
                 'pp-cinder-surge-i-v1', 'pp-cinder-surge-ii-v1'}
STORMCALLER_IDS = {'pp-breachfire-v1', 'pp-tempest-sigil-v1',
                   'pp-stormbreakers-wrath-v1', 'pp-tempest-barrage-v1',
                   'pp-tempest-rising-v1'}

def pp_character(move):
    """Route identity changes only artwork and accent, never qualification."""
    if move['id'] in RUNESMITH_IDS:
        return 'Runesmith', 'Orangutan · Runesmith', 'runesmith'
    if move['id'] in STORMCALLER_IDS:
        return 'Stormcaller', 'Chimp · Stormcaller', 'stormcaller'
    return 'Arcanist', 'Gorilla · Arcanist', 'arcanist'

def display_families(moves):
    """Keep the strongest matching definition per visible family; Inspect retains all."""
    families = {}
    for move in sorted(moves, key=pp_move_rank, reverse=True):
        families.setdefault(move.get('display_family', move.get('family', move['id'])), move)
    return list(families.values())

def _rule_implies(child, parent):
    """Only literal rule nesting, never correlation or historical-cohort overlap."""
    for field, op, cutoff in parent['entry'] + parent['conditions']:
        def enough(check):
            f, comparison, value = check
            if f != field:
                return False
            if op == '>=':
                return comparison in ('>=', '==') and value >= cutoff
            if op == '<=':
                return comparison in ('<=', '<', '==') and value <= cutoff
            if op == '<':
                return comparison == '<' and value <= cutoff or comparison in ('<=', '==') and value < cutoff
            return comparison == '==' and value == cutoff
        if not any(enough(check) for check in child['entry'] + child['conditions']):
            return False
    return True

def display_routes(moves):
    """Display only: stronger levels/combos cover their weaker stacked steps."""
    candidates = display_families(moves)
    result = []
    for index, move in enumerate(candidates):
        # Earlier records are the already established stronger display order.
        if any(_rule_implies(stronger, move) for stronger in candidates[:index]):
            continue
        result.append(move)
    return result

def _matches(value, op, cutoff):
    if value is None:
        return False
    return {'>=': value >= cutoff, '<=': value <= cutoff,
            '<': value < cutoff, '==': value == cutoff}[op]

def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

@lru_cache(maxsize=1)
def records():
    return json.loads(Path(__file__).with_name('pp_card_records.json').read_text(encoding='utf-8'))

def onice_xg(row):
    value = _num(row.get('PP_OnIce_xG_per_game'))
    if value is None:
        rate, minutes = _num(row.get('PP_OnIce_xGF60')), _num(row.get('PP_TOI_per_game'))
        if rate is not None and minutes is not None and rate >= 0 and minutes >= 0:
            value = round(rate * minutes / 60, 3)
    return value

def fired_pp_moves(row, line=.5):
    """Each lane uses its own approved entry; no universal PP1/Matrix/confidence gate."""
    xg = onice_xg(row)
    if _num(line) != .5 or row.get('Model_Stats_Season') != '2026-2027' or xg is None:
        return []
    valid_context = (row.get('PP_Card_Context_Version') == CONTEXT_VERSION
                     and str(row.get('PP_Card_Context_AsOf')) == str(row.get('Date'))[:10])
    fired = []
    for move in records()['moves']:
        lower = move.get('lane') == 'layer1-lower-xg'
        lower_current = (row.get('PP_Lower_Context_Version') == LOWER_CONTEXT_VERSION
                         and str(row.get('PP_Lower_Context_AsOf')) == str(row.get('Date'))[:10])
        if lower and not (valid_context and lower_current):
            continue
        if not lower and move['conditions'] and not valid_context:
            continue
        matches = True
        for field, op, cutoff in move['entry'] + move['conditions']:
            value = (_num(line) if field == 'PPP line' else xg if field == 'PP_OnIce_xG_per_game'
                     else _num(row.get(field)))
            if not _matches(value, op, cutoff):
                matches = False
                break
        if matches:
            fired.append(move)
    return sorted(fired, key=lambda m: (-m['record']['rate'], -m['record']['plays'], m['id']))

def rank_pp_cards(frame, quotes):
    """One player-game card, using the existing best-quote selection and view filters."""
    rows = {(str(r.get('Player')), str(r.get('Team')), str(r.get('Game'))): r
            for r in frame.to_dict('records')}
    cards = {}
    for quote in quotes.to_dict('records'):
        key = tuple(str(quote.get(k)) for k in ('Player', 'Team', 'Game'))
        row = rows.get(key)
        odds = _num(quote.get('Over odds'))
        if row is None or odds is None or odds == 0:
            continue
        moves = fired_pp_moves(row, quote.get('PPP line'))
        if not moves:
            continue
        moves = sorted(moves, key=pp_move_rank, reverse=True)
        card = dict(row=row, quote=quote, moves=moves, best=moves[0], xg=onice_xg(row),
                    be=100*(100/(odds+100) if odds > 0 else -odds/(-odds+100)))
        if key not in cards or odds > cards[key]['quote']['Over odds']:
            cards[key] = card
    return sorted(cards.values(), key=lambda c: (pp_move_rank(c['best']),
                   c['quote']['Player']), reverse=True)

def _h(value):
    return escape(str(value), quote=True)

def _value(value, digits=2):
    value = _num(value)
    return 'Unavailable' if value is None else f'{value:.{digits}f}'

def _rule_checks(row, move, xg):
    checks = []
    for field, op, cutoff in move['entry'] + move['conditions']:
        if field == 'PPP line':
            continue
        label, unit = LABELS[field]
        actual = xg if field == 'PP_OnIce_xG_per_game' else row.get(field)
        checks.append(f'{_h(label)} <b>{_value(actual,3)}{unit}</b> '
                      f'{dict([(">=","≥"),("<=","≤"),("<","<"),("==","=")])[op]}{cutoff:g}{unit}')
    return '<br>'.join(checks)

def render_pp_cards(cards, portrait_uri='', form_renderer=None):
    """Compact front; dated PPP form and distinct routes are inside Inspect."""
    articles = []
    for index, card in enumerate(cards, 1):
        row, quote, move = card['row'], card['quote'], card['best']
        record = move['record']
        variant, character, accent = pp_character(move)
        portrait = portrait_uri.get(variant, '') if isinstance(portrait_uri, dict) else portrait_uri
        families = display_routes(card['moves'])
        odds = float(quote['Over odds'])
        matches = []
        nested_matches = []
        family_ids = {fired['id'] for fired in families}
        for fired in card['moves']:
            r = fired['record']
            target = matches if fired['id'] in family_ids else nested_matches
            target.append(f'<section class="pp-w-move" data-move-id="{_h(fired["id"])}">'
                f'<b>{_h(fired["name"])}</b>{grade_badges(fired)} <span>{_h(fired["kind"])}</span>'
                f'<p><strong>{r["hits"]}/{r["plays"]} · {r["rate"]:.1f}%</strong> · '
                f'sample book BE {r["BE"]:.1f}% · +{r["gap"]:.1f} points · '
                f'{r["players"]} players · {r["dates"]} slate dates</p>'
                f'<div>{_rule_checks(row, fired, card["xg"])}</div>{pp_return_panel(fired,compact=False)}</section>')
        source = []
        for side, window in [('Team', 'L5'), ('Opp', 'L5'), ('Opp', 'L10'), ('Opp', 'Season')]:
            prefix = f'PP_Card_{side}_{window}'
            source.append(f'{side} {window}: {_value(row.get(prefix+"_GP"), 0)} games'
                          f' · through {_h(row.get(prefix+"_Last_Date") or "unavailable")}')
        if move.get('lane') == 'layer1-lower-xg':
            for label, field, dayfield in [
                ('Player season', 'PP_Lower_Player_Season_GP', 'PP_Lower_Player_Last_Date'),
                ('Team L10 credits', 'PP_Lower_Team_L10_GP', 'PP_Lower_Team_Last_Date'),
                ('Opponent L10 credits', 'PP_Lower_Opp_L10_GP', 'PP_Lower_Opp_Last_Date')]:
                source.append(f'{label}: {_value(row.get(field),0)} games · through '
                              f'{_h(row.get(dayfield) or "unavailable")}')
        form = ppp_l10_chart(row)
        image = (f'<img class="pp-w-portrait" src="{_h(portrait)}" alt="{_h(character)}" />'
                 if portrait else '<div class="pp-w-portrait pp-w-sigil" aria-hidden="true">ϟ</div>')
        upgraded = ' pp-w-combo' if move['kind'] == 'COMBO SPELL' else ''
        performance = move_grade(move)
        graded = (' rg-card-' + performance['tier'].lower() + (' rg-card-watch' if performance['watch'] else '')) if performance else ''
        nested = (f'<details class="pp-w-all-definitions"><summary>Other matched levels ({len(nested_matches)})</summary>{"".join(nested_matches)}</details>' if nested_matches else '')
        articles.append(f'<article class="pp-w-card pp-w-{accent}{upgraded}{graded}"><div class="pp-w-front">'
            f'<div class="pp-w-body"><div class="pp-w-heading"><span>{index:02d}</span>{grade_badges(move)} '
            f'<strong>{_h(quote["Player"])}</strong> <small>{_h(quote["Game"])}</small></div>'
            f'<div class="pp-w-name">{_h(move["name"])} <small>{_h(move["kind"])}</small></div>'
            f'<div class="pp-w-character">{_h(character)}</div>'
            f'<p class="pp-w-price">OVER 0.5 POWER-PLAY POINTS <b>{odds:+.0f}</b> · {_h(quote["Book"])}</p>'
            f'<p>BOOK BREAK-EVEN <b>{card["be"]:.1f}%</b></p>'
            f'<p class="pp-w-quick">PP on-ice xG/game <b>{card["xg"]:.3f}</b> · '
            f'PP <b>{_value(quote.get("PP TOI/game"), 1)} min/game</b> · {_h(quote.get("PP unit", "Unit unknown"))}'
            f' · <b>{len(families)}</b> route {"family" if len(families) == 1 else "families"}</p>'
            f'<p class="pp-w-l10-quick">PPP L10 <b>{_h(ppp_l10_label(row))}</b> · dated chart in Inspect</p>{pp_return_panel(move)}</div>'
            f'<div class="pp-w-record"><strong>{record["rate"]:.1f}%</strong>'
            f'<b>{record["hits"]}/{record["plays"]}</b><small>V1 MOVE %</small></div>{image}</div>'
            f'<details class="pp-w-inspect"><summary>Inspect · PPP L10, {len(families)} distinct {"route" if len(families) == 1 else "routes"} &amp; matchup</summary>{form}'
            f'<div class="pp-w-info">Move sample book BE <b>{record["BE"]:.1f}%</b> · '
            f'<b>{record["gap"]:+.1f} points</b> historical gap<br>'
            'One player-game counts once in each matching rule. Nested and overlapping moves are not independent evidence.</div>'
            f'{"".join(matches)}{nested}<div class="pp-w-info"><b>Official pregame windows</b><br>{"<br>".join(source)}'
            f'<br>As of {_h(row.get("Date"))} · all PP/PK strengths; last up-to-5/10 completed regular-season games.</div>'
            f'<div class="pp-w-info">Lineup: {_h(row.get("Lineup_Status", "Pending"))}<br>'
            f'2026–27 research samples: higher-xG Oct 17–Nov 5; Layer 1 lower-xG Oct 16–Nov 6 · {VERSION}. '
            'Historical move results are not tonight’s PPP probabilities. On-ice xG describes team chances while this player is on the ice.</div>'
            '</details></article>')
    return '<style>' + FORM_CSS + RETURN_CSS + GRADE_CSS + '''</style><style>
    .pp-w{container-type:inline-size;color:#eff6ff;font-family:Arial,sans-serif;background:#0d1525;border:1px solid #71649c;border-radius:14px;padding:12px}
    .pp-w *{box-sizing:border-box}.pp-w-title{color:#e6d0ff;font-size:21px;margin:4px 0 7px}.pp-w-caption{font-size:13px;color:#c8d7f1;line-height:1.5;margin-bottom:12px}
    .pp-w-card{position:relative;border:1px solid #586783;border-left:4px solid #b596ff;border-radius:9px;background:linear-gradient(115deg,#27223e,#1c293c);padding:12px;margin:9px 0;overflow:hidden}
    .pp-w-runesmith{border-left-color:#efc267;background:linear-gradient(115deg,#30293a,#172c43)}.pp-w-stormcaller{border-left-color:#6ae3e1;background:linear-gradient(115deg,#193b42,#20233e)}
    .pp-w-character{font-size:12px;font-weight:750;color:#d2c2f5;margin:4px 0 7px}.pp-w-runesmith .pp-w-character{color:#f1d8a0}.pp-w-stormcaller .pp-w-character{color:#a4f3ef}
    .pp-w-family-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.pp-w-family-list span{border:1px solid #72658e;background:#1d2440;color:#eedbff;border-radius:5px;padding:5px 8px;font-size:12px;font-weight:700}
    .pp-w-front{display:grid;grid-template-columns:minmax(0,1fr) 90px 110px;gap:12px;align-items:start}.pp-w-body{min-width:0}
    .pp-w-heading{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}.pp-w-heading>strong{font-size:18px}.pp-w-heading>span{color:#f4cd80;font-size:12px;font-weight:800}.pp-w-heading small{font-size:12px;color:#c8d7f1}
    .pp-w-name{font-size:16px;color:#efd095;font-weight:850;margin:8px 0}.pp-w-name small{font-size:10px;border:1px solid #b79ce9;border-radius:3px;padding:3px;margin-left:5px;color:#e7daff;white-space:nowrap}
    .pp-w-card p{font-size:13px;line-height:1.55;margin:6px 0}.pp-w-card .pp-w-price{font-size:14px;background:#111c2f;border:1px solid #4b617b;border-radius:6px;padding:8px}.pp-w-price b{color:#ffe3a0;font-size:17px}
    .pp-w-quick{color:#d9e8fa}.pp-w-record{text-align:right;display:flex;flex-direction:column;gap:4px}.pp-w-record strong{font-size:28px;color:#fff}.pp-w-record b{font-size:15px;color:#f5d89c}.pp-w-record small{font-size:9px;color:#bfcce2}
    .pp-w-l10-quick{grid-column:1/-1;color:#cbdaf0;font-size:12px!important}.pp-w-l10-quick b{color:#fff}
    .pp-w-portrait{width:110px;height:145px;object-fit:cover;object-position:center top;border:1px solid #b59ae7;border-radius:8px;background:#10182b}.pp-w-sigil{display:grid;place-items:center;font-size:90px;color:#c4a5ff}
    .pp-w-inspect{border-top:1px solid #4a5971;margin-top:10px;padding-top:4px;font-size:13px}.pp-w-inspect>summary{cursor:pointer;color:#f0d59b;font-weight:750;padding:5px 0}.pp-w-move,.pp-w-info{background:#101b2c;border:1px solid #44546e;border-radius:7px;padding:10px;margin-top:8px;line-height:1.6;font-size:13px}.pp-w-move>b{color:#f1dba7;font-size:15px}.pp-w-move>span{float:right;color:#c8aef4;font-size:10px}.pp-w-info{color:#cbdaf0}
    .pp-w-combo{border-color:#b7a0f0;box-shadow:inset 3px 0 18px #985bce30,0 0 9px #b38dff32}.pp-w-combo .pp-w-name{text-shadow:0 0 10px #ba8eff88}
    .pp-w-combo:before{content:'';position:absolute;pointer-events:none;inset:0;background:linear-gradient(110deg,transparent 30%,#bb91ff12 50%,transparent 70%);animation:pp-w-shimmer 6s infinite}
    @keyframes pp-w-shimmer{0%,30%{transform:translateX(-100%)}80%,100%{transform:translateX(100%)}}@media(prefers-reduced-motion:reduce){.pp-w-combo:before{animation:none}}
    .pp-w .wn-form{margin-top:10px;font-size:13px}.pp-w .wn-form summary{cursor:pointer;color:#f0d59b}.pp-w .wn-form-tiles{display:flex;flex-wrap:wrap;gap:6px}.pp-w .wn-form-tiles>div{background:#101b2c;padding:8px;border-radius:5px;min-width:110px}.pp-w .wn-form-tiles small,.pp-w .wn-form-tiles strong{display:block}.pp-w .wn-form-meta{margin:7px 0}.pp-w .wn-form-strip{display:flex;flex-wrap:wrap;gap:4px}.pp-w .wn-form-strip>span{padding:7px;border-radius:5px}.pp-w .wn-form-strip .hit{background:#245744}.pp-w .wn-form-strip .miss{background:#663540}.pp-w .wn-form-strip b,.pp-w .wn-form-strip small{display:block;text-align:center}.pp-w .wn-form-strip small{font-size:11px;margin-top:4px}
    @container(max-width:600px){.pp-w-front{grid-template-columns:minmax(0,1fr) 75px}.pp-w-portrait{display:none}.pp-w-record strong{font-size:25px}.pp-w-heading>strong{font-size:17px}.pp-w-name small{display:inline-block;margin:4px 0}.pp-w-price{overflow-wrap:anywhere}}
    </style><section class="pp-w"><h3 class="pp-w-title">ϟ Power Play · Spellbook</h3><div class="pp-w-caption">Strongest return-ranked move leads: S / A / B, then recent status and total ROI. Watch flags recent net losses. Inspect keeps every matched route and level.</div>''' + ''.join(articles) + '</section>'
