"""Independent TOI20 Goal path with highest-qualified PPP-drought enhancement."""

from goal_moves_current import _number
from goal_star_moves import _record


ENTRY = "Green + Goal 0.5 + StarScore >= 84.5 + TOI/game >= 20 + L10 shots <= 39"
BASE_RECORD = (66, 131, 39.0, 29, 18)
# The historical samples use >= thresholds. Only the highest level is displayed.
LEVELS = ((1, "I", (45, 79, 39.4, 27, 17)),
          (2, "II", (35, 53, 39.6, 22, 17)))


def _make(key, name, kind, rule, record, level=None):
    move = _record(key, name, kind, "smash.svg", rule, record)
    move.update(id=f"goal_toi20:{key}", route_family="star_toi20", level=level)
    return move


def toi_goals_moves(row):
    line = _number(row, "Goal_Line")
    if line is None:
        line = _number(row, "ATG_Line")
    star = _number(row, "StarScore")
    minutes = _number(row, "TOI_per_game")
    shots = _number(row, "L10_S")
    if (line != .5 or str(row.get("Matrix_Goal", "")).strip().casefold() != "green"
            or str(row.get("Model_Stats_Season", "")) != "2026-2027"
            or str(row.get("Form_Season", "")) != "2026-27"
            or star is None or star < 84.5 or minutes is None or minutes < 20
            or shots is None or shots < 0 or shots > 39):
        return []
    moves = [_make("base", "Star Barrage", "BASE ATTACK", ENTRY, BASE_RECORD)]
    drought = _number(row, "Drought_PPP")
    if drought is not None:
        for cutoff, label, record in reversed(LEVELS):
            if drought >= cutoff:
                moves.append(_make(f"ppp_{label.lower()}", f"Runic Barrage {label}",
                                   "SPECIAL ATTACK", ENTRY + f" + PPP drought >= {cutoff}",
                                   record, level=label))
                break
    return moves
