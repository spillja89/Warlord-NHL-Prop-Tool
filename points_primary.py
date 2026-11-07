"""Resolve Points primary lines from book markets, independently of model scores."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import median

VERSION = "points-primary-market-2026-11-07-v1"


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def valid_quote(line, odds):
    return (line is not None and line >= .5 and abs(line % 1 - .5) < 1e-8
            and odds is not None and abs(odds) >= 100)


def implied(odds):
    return -odds / (100 - odds) if odds < 0 else 100 / (100 + odds)


def select_primary(by_line, main_quotes=()):
    """Book main-market consensus; price-balanced inference if markers are absent.

    Each book gets one vote. Repeated/alternate quotes cannot inflate support.
    A line inferred from ladder prices is explicitly labeled as inferred.
    """
    lines = {float(line): quote for line, quote in by_line.items()
             if valid_quote(number(line), number(quote[0]))}
    if not lines:
        return None, "Unavailable", []
    books = defaultdict(list)
    for quote in main_quotes:
        line, odds = number(quote.get("line")), number(quote.get("odds"))
        book = str(quote.get("book") or "").strip()
        if book and line in lines and valid_quote(line, odds):
            books[book.casefold()].append((abs(implied(odds) - .5), line, odds, book))
    votes = defaultdict(list)
    for quotes in books.values():
        _, line, odds, book = min(quotes)
        votes[line].append((odds, book))
    if votes:
        chosen = min(votes, key=lambda line: (
            -len(votes[line]), median(abs(implied(o) - .5) for o, _ in votes[line]), line))
        return chosen, "Book main-market consensus", sorted(book for _, book in votes[chosen])
    chosen = min(lines, key=lambda line: (abs(implied(float(lines[line][0])) - .5), line))
    return chosen, "Inferred from available ladder prices", []


def resolve_frame(frame):
    """Repair selected lines from saved observed quotes; never create a new rung.

    Does not change player/matchup scores, mu, rules or historical move records.
    Line-specific probability/EV fields are refreshed if the selected quote changes.
    """
    out = frame.copy()
    for idx, row in out.iterrows():
        if row.get("Model_Stats_Season") != "2026-2027":
            continue
        quotes = {}
        def add(line, odds, book):
            line, odds = number(line), number(odds)
            book = str(book or "").strip()
            if valid_quote(line, odds) and book and book.casefold() not in {"nan", "none"}:
                if line not in quotes or odds > quotes[line][0]:
                    quotes[line] = (odds, book)
        add(row.get("Points_Line"), row.get("Points_Odds_Over"), row.get("Points_Book"))
        for suffix in ("", "_1", "_2", "_3", "_4"):
            add(row.get("BDL_Points_Line" + suffix), row.get("BDL_Points_Odds" + suffix), row.get("BDL_Points_Book" + suffix))
        try:
            mains = json.loads(row.get("Points_Main_Market_Quotes") or "[]")
            if not isinstance(mains, list):
                mains = []
        except (TypeError, ValueError):
            mains = []
        line, source, books = select_primary(quotes, mains)
        if line is None:
            continue
        odds, book = quotes[line]
        old_line = number(row.get("Points_Line"))
        changed = old_line != line or number(row.get("Points_Odds_Over")) != odds
        if old_line != line and number(row.get("Points_PrePrimary_Line")) is None:
            for field in ("Line", "Odds_Over", "Book"):
                out.at[idx, "Points_PrePrimary_" + field] = row.get("Points_" + field)
        for field, value in (("Line", line), ("Odds_Over", odds), ("Book", book)):
            out.at[idx, "Points_" + field] = value
        out.at[idx, "Points_Primary_Line"] = line
        out.at[idx, "Points_Primary_Source"] = source
        out.at[idx, "Points_Primary_Books"] = json.dumps(books)
        out.at[idx, "Points_Primary_Version"] = VERSION
        if not changed:
            continue
        # Preserve mu and use the exact existing distribution for the new line.
        from odds_ev_bdl import _model_prob_over, _compute_ev_pct
        mu = number(row.get("Points_mu"))
        probability = _model_prob_over("Points", mu, line) if mu is not None and mu >= 0 else None
        out.at[idx, "Points_p_model_over"] = probability
        out.at[idx, "Points_Model%"] = round(100 * probability, 1) if probability is not None else None
        out.at[idx, "Points_p_imp_over"] = implied(odds)
        ev = _compute_ev_pct(probability, odds) if probability is not None else None
        out.at[idx, "Points_EVpct_over"] = ev
        out.at[idx, "Points_EV%"] = round(ev, 1) if ev is not None else None
        tonight_mu = number(row.get("Tonight_Points_mu"))
        if tonight_mu is not None:
            import pandas as pd
            from tonight_goal_projection import _nb_over_pct
            pct = float(_nb_over_pct(pd.Series([tonight_mu]), pd.Series([line]), .35).iloc[0])
            out.at[idx, "Tonight_Points_over_pct"] = round(pct, 1)
            if "Tonight_Points_vs_Book_pp" in out:
                out.at[idx, "Tonight_Points_vs_Book_pp"] = round(pct - 100 * implied(odds), 1)
        elif old_line != line:
            # A probability for the former rung must not follow the new line.
            for field in ("Tonight_Points_over_pct", "Tonight_Points_vs_Book_pp"):
                if field in out:
                    out.at[idx, field] = None
    return out
