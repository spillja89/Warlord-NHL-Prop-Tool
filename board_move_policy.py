"""Board display patch: current combined move record must round to 50% or higher.

This only selects visible cards. Frozen rules, captures, returns and saved picks
remain unchanged. Exact matching return identities prevent old-season fallback.
"""
from decimal import Decimal, InvalidOperation

VERSION = 'board-display-v2-2026-11-27'
MIN_HIT_RATE_PCT = Decimal('49.5')


def current_record(move):
    if not isinstance(move, dict):
        return None
    # Legacy Points has a verified current-season replay. Never use its separate
    # 2025-26 seed as the headline or to bypass the current-season floor.
    from points_return_monitor import record_for as points_record
    row = points_record(move)
    if row:
        return row
    from move_return_grades import return_record
    return return_record(move)


def current_stats(move):
    row = current_record(move)
    return row.get('full') if row else None


def meets_floor(stats):
    if not isinstance(stats, dict):
        return False
    try:
        hits, plays = (Decimal(str(stats.get(k))) for k in ('hits', 'plays'))
        if not all(v.is_finite() and v == v.to_integral_value() for v in (hits, plays)):
            return False
        return plays > 0 and 0 <= hits <= plays and hits * 100 >= plays * MIN_HIT_RATE_PCT
    except (InvalidOperation, TypeError, ValueError):
        return False


def board_eligible(move):
    if not isinstance(move, dict):
        return False
    if any(move.get(k) for k in ('goal_retired', 'goal_archived', 'sog_retired', 'suppress_badge')):
        return False
    return meets_floor(current_stats(move))
