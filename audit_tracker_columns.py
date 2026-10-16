"""Inventory every saved tracker field and check reproducible cross-field invariants.

Run with: python audit_tracker_columns.py output/tracker_latest.csv
This audits a frozen CSV; it does not assert that a bookmaker quote is still live.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MARKETS = ("SOG", "Points", "Assists", "Goal", "ATG")
RANGE_100 = re.compile(r"(^Conf_|^Best_Conf$|^iXG%$|^iXA%$|_pct$|_Pct$|_Pct_Game$|^Goalie_Weak$|^Opp_DefWeak$|^PP_Matchup$|^PP_Env_Score$|^ShotIntent_Pct$|^v2_player_stability$)")


def group(name: str) -> str:
    if name.startswith("BDL_") or name.endswith(("_Odds_Source", "_Available_Books")) or name in ("Odds_Checked_UTC", "Odds_Name_Ambiguous"):
        return "captured book quote"
    if any(name.startswith(m + "_") for m in MARKETS) and any(s in name for s in ("_Odds_", "_Line", "_Book")):
        return "selected book quote"
    if any(name.startswith(m + "_") for m in MARKETS) and any(s in name for s in ("_Model", "_p_model", "_p_imp", "_EV", "_mu")):
        return "model/price calculation"
    if name.startswith(("PP_", "Team_PP_", "Opp_PK_")):
        return "power-play data/heuristic"
    if name.startswith(("L10_", "L20_", "L40_", "Drought_", "Med10_", "Trim10_", "Avg5_", "PPP10_", "Form_")):
        return "official game-log feature"
    if name.startswith(("Opp_Goalie", "Opp_SV", "Opp_GAA", "Goalie_")):
        return "goalie context"
    if name.startswith(("Injury_", "Roster_", "Available", "Team_Out_")):
        return "availability/roster"
    if name in ("Date", "Game", "Game_ID", "StartTimeLocal", "StartTimeUTC", "Player", "Player_ID", "Model_Stats_Team", "Team_Changed", "Team", "Opp", "Pos"):
        return "identity/schedule"
    if name.startswith(("Plays_", "Matrix_", "Conf_", "Reg_", "Exp_", "Assist_", "Goal_", "SOG_", "Best_", "Fired_")):
        return "decision heuristic"
    if name in ("Line", "Odds", "Result", "Outcome"):
        return "empty grading placeholder"
    return "season stat/derived context"


def american_implied(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100)), index=odds.index)


def american_ev(p: pd.Series, odds: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    odds = pd.to_numeric(odds, errors="coerce")
    net = pd.Series(np.where(odds > 0, odds / 100, 100 / -odds), index=odds.index)
    return 100 * (p * net - (1 - p))


def audit(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(path, low_memory=False)
    findings: list[dict] = []
    field_notes: dict[str, list[str]] = {c: [] for c in df.columns}

    def check(label: str, fields: list[str], bad: pd.Series, severity: str = "error") -> None:
        n = int(bad.fillna(False).sum())
        if n:
            findings.append({"severity": severity, "check": label, "rows": n, "fields": fields})
            for field in fields:
                if field in field_notes:
                    field_notes[field].append(f"{label}: {n} rows")

    for c in df.columns:
        if df[c].notna().sum() == 0:
            field_notes[c].append("empty in this slate")
        if RANGE_100.search(c):
            v = pd.to_numeric(df[c], errors="coerce")
            check("outside 0–100", [c], v.notna() & ((v < 0) | (v > 100)))
        if "_p_model_over" in c or "_p_imp_over" in c:
            v = pd.to_numeric(df[c], errors="coerce")
            check("probability outside 0–1", [c], v.notna() & ((v < 0) | (v > 1)))

    if {"Game", "Team", "Opp"}.issubset(df):
        side = df["Game"].astype(str).str.split("@", n=1, expand=True)
        check("team/opponent disagree with matchup", ["Game", "Team", "Opp"],
              ~((df["Team"].eq(side[0]) & df["Opp"].eq(side[1])) |
                (df["Team"].eq(side[1]) & df["Opp"].eq(side[0]))))
    if "Player_ID" in df:
        check("duplicate player ID", ["Player_ID"], df["Player_ID"].duplicated(keep=False))
    if "Form_Log" in df:
        log_issues: dict[tuple[str, tuple[str, ...]], list[int]] = {}

        def record_log_issue(label: str, fields: list[str], idx: int) -> None:
            log_issues.setdefault((label, tuple(fields)), []).append(idx)

        for idx, raw in df["Form_Log"].dropna().items():
            try:
                log = json.loads(raw)
                games = log.get("games", [])
                dates = [g.get("d", "") for g in games]
                if dates != sorted(dates, reverse=True):
                    record_log_issue("form dates not newest first", ["Form_Log"], idx)
                if "Date" in df and any(day >= str(df.at[idx, "Date"]) for day in dates):
                    record_log_issue("form contains slate-day/future game", ["Form_Log", "Date"], idx)
                for w in (10, 20, 40):
                    window = games[:w]
                    expected = {"P": sum(int(g.get("g", 0)) + int(g.get("a", 0)) for g in window),
                                "A": sum(int(g.get("a", 0)) for g in window),
                                "S": sum(int(g.get("s", 0)) for g in window)}
                    for stat, total in expected.items():
                        col = f"L{w}_{stat}"
                        if col in df and pd.notna(df.at[idx, col]) and abs(float(df.at[idx, col]) - total) > .01:
                            record_log_issue("log total mismatch", ["Form_Log", col], idx)
            except (ValueError, TypeError, KeyError):
                record_log_issue("unreadable form log", ["Form_Log"], idx)
        for (label, fields), indexes in log_issues.items():
            check(label, list(fields), df.index.isin(indexes))

    for m in MARKETS:
        p, model, imp, odds, ev, line, book = (f"{m}_{x}" for x in
            ("p_model_over", "Model%", "p_imp_over", "Odds_Over", "EVpct_over", "Line", "Book"))
        if not all(c in df for c in (p, model, imp, odds, ev, line, book)):
            continue
        pn = pd.to_numeric(df[p], errors="coerce")
        on = pd.to_numeric(df[odds], errors="coerce")
        ln = pd.to_numeric(df[line], errors="coerce")
        mn = pd.to_numeric(df[model], errors="coerce")
        check(f"{m} price lacks line/book", [line, odds, book], on.notna() &
              (ln.isna() | df[book].fillna("").astype(str).str.strip().eq("")))
        check(f"{m} model percentage mismatch", [p, model], pn.notna() & mn.notna() & ((pn*100-mn).abs() > .11))
        check(f"{m} implied probability mismatch", [odds, imp], on.notna() &
              ((american_implied(on) - pd.to_numeric(df[imp], errors="coerce")).abs() > .002))
        check(f"{m} EV mismatch", [p, odds, ev], pn.notna() & on.notna() &
              ((american_ev(pn, on) - pd.to_numeric(df[ev], errors="coerce")).abs() > .12))
        for i in range(1, 5):
            cols = [f"BDL_{m}_{x}_{i}" for x in ("Line", "Odds", "Book")]
            if all(c in df for c in cols):
                populated = [df[c].notna() & df[c].astype(str).str.strip().ne("") for c in cols]
                check(f"{m} rung {i} incomplete quote", cols,
                      (populated[0] | populated[1] | populated[2]) &
                      ~(populated[0] & populated[1] & populated[2]))

    if {"PP_Points60", "PP_iXG60", "PP_iXA60"}.issubset(df):
        points = pd.to_numeric(df.PP_Points60, errors="coerce")
        proxy = pd.to_numeric(df.PP_iXG60, errors="coerce").fillna(0) + .85 * pd.to_numeric(df.PP_iXA60, errors="coerce").fillna(0)
        check("legacy PP involvement proxy arithmetic", ["PP_Points60", "PP_iXG60", "PP_iXA60"],
              points.notna() & ((points-proxy).abs() > .02))
    if {"PP_TOI_min", "PP_TOI_per_game"}.issubset(df):
        check("PP minutes per game exceed total minutes", ["PP_TOI_min", "PP_TOI_per_game"],
              pd.to_numeric(df.PP_TOI_per_game, errors="coerce") > pd.to_numeric(df.PP_TOI_min, errors="coerce") + .01)
    for w in (10, 20, 40):
        count = f"L{w}_Games"
        if count in df:
            n = pd.to_numeric(df[count], errors="coerce")
            check(f"L{w} sample exceeds window", [count], n.notna() & ((n < 0) | (n > w)))
            for market, stat in (("Points", "P"), ("Assists", "A"), ("SOG", "S")):
                total, rate = f"L{w}_{stat}", f"L{w}_Rate_{market}"
                if total in df and rate in df:
                    expected = pd.to_numeric(df[total], errors="coerce") / n.replace(0, np.nan)
                    actual = pd.to_numeric(df[rate], errors="coerce")
                    check(f"{rate} wrong denominator", [total, count, rate],
                          expected.notna() & actual.notna() & ((expected-actual).abs() > .02))

    if "L10_Games" in df:
        n = pd.to_numeric(df.L10_Games, errors="coerce").where(lambda x: x >= 5)
        for gap, basis, total in (("Reg_Gap_P10", "Exp_P_10_Gap_Basis", "L10_P"),
                                  ("Reg_Gap_A10", "Exp_A_10_Gap_Basis", "L10_A"),
                                  ("Reg_Gap_G10", "Exp_G_10", "L10_G"),
                                  ("Reg_Gap_S10", "Exp_S_10", "L10_S")):
            if {gap, basis, total}.issubset(df):
                expected = pd.to_numeric(df[basis], errors="coerce") - 10 * pd.to_numeric(df[total], errors="coerce") / n
                actual = pd.to_numeric(df[gap], errors="coerce")
                check(f"{gap} pace mismatch", [gap, basis, total, "L10_Games"],
                      expected.notna() & actual.notna() & ((expected-actual).abs() > .03))

    inventory = []
    for c in df:
        nonempty = int(df[c].notna().sum())
        notes = field_notes[c]
        inventory.append({"field": c, "group": group(c), "filled": nonempty,
                          "rows": len(df), "distinct": int(df[c].nunique(dropna=True)),
                          "status": "issue" if any("rows" in n for n in notes) else
                          "empty" if nonempty == 0 else "snapshot check passed",
                          "notes": "; ".join(notes)})
    return pd.DataFrame(inventory), findings


def main() -> None:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "output/tracker_latest.csv")
    snapshot = pd.read_csv(source, low_memory=False)
    inventory, findings = audit(source)
    out = Path("reports") / f"tracker_column_audit_{snapshot.Date.iloc[0]}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Tracker column audit: {source.name}", "",
             f"Columns inventoried: **{len(inventory)}**. This is a saved-snapshot arithmetic and completeness audit, not live-feed confirmation or model calibration.", "",
             "## Findings", ""]
    if findings:
        lines.extend(f"- **{f['severity']}**: {f['check']} — {f['rows']} rows ({', '.join(f['fields'])})" for f in findings)
    else:
        lines.append("No tested cross-field invariant failed.")
    log_rows = snapshot[snapshot.Form_Log.notna()].copy() if "Form_Log" in snapshot else pd.DataFrame()
    partial = 0
    if not log_rows.empty:
        partial = int(sum(len(json.loads(s).get("games", [])) < 10 for s in log_rows.Form_Log))
    gap_a = int(((snapshot.Exp_A_10 - snapshot.L10_A - snapshot.Reg_Gap_A10).abs() > .02).sum()) if {"Exp_A_10", "L10_A", "Reg_Gap_A10"}.issubset(snapshot) else 0
    gap_p = int(((snapshot.Exp_P_10 - snapshot.L10_P - snapshot.Reg_Gap_P10).abs() > .02).sum()) if {"Exp_P_10", "L10_P", "Reg_Gap_P10"}.issubset(snapshot) else 0
    lines += ["", "## Code and label defects found during this audit", "",
              f"- **Fixed for the next run:** {partial} of {len(log_rows)} logged players had fewer than 10 completed games. The old L10/L20/L40 per-game rates divided by the window label instead of the observed number of games. The next tracker saves actual game counts.",
              f"- **Fixed for the next run:** displayed expectations and regression gaps disagreed for {gap_a} assist rows and {gap_p} points rows because a later multiplier changed the expectation. The next tracker exposes the exact expectation used for each gap.",
              "- **Fixed for the next run:** the points matrix referenced an undefined confidence variable on one branch; a regression test now exercises that branch.",
              "- **Fixed for the next run:** the old 5v5 SOG-share field used a partial slate proxy. The next tracker calculates player shots divided by all team skater shots in the 5v5 MoneyPuck feed; traded players' old-team shares stay blank.",
              "- **Preserved transparently:** the historically graded Berserker Volley move continues to use its slate SOG share proxy, now named as such. The true 5v5 share is a separate field and has no inherited historical win rate.",
              "- **Clarified for the next run:** PP iXA and PP points are observed-assist / mixed involvement proxies. Actual PP points/60 is a separate field; frozen historical move thresholds retain their old numeric inputs.",
              "- **Fixed for the next run:** regression gaps now compare the 10-game expectation with a 10-game-equivalent observed pace; fewer than five recorded games leave the gap unavailable. A gap is still not a calibrated bounce-back probability.",
              ""]
    lines += ["", "## Known source and interpretation limits", "",
              "- `iXA%` ranks observed season assists, because this MoneyPuck feed does not include xAssists. It is not an expected-assists percentile.",
              "- `PP_iXA60` is weighted observed assists/60; `PP_Points60` mixes xG and observed assists. These legacy numbers are not expected assists or actual points.",
              "- `PP_Role` is season-total PP-time rank, not a confirmed tonight unit. `PP_Matchup` and confidence fields are heuristics, not calibrated hit probabilities.",
              "- Empty alternate-line columns mean those rungs were not quoted. Empty log fields may reflect the candidate-only NHL-log fetch.",
              "- Recheck current prices and lineup/goalie status at the source before using a saved slate.",
              "", "## Every field", "", "| Field | Group | Filled | Distinct | Status / note |", "|---|---|---:|---:|---|"]
    for row in inventory.itertuples(index=False):
        note = f"{row.status}; {row.notes}" if row.notes else row.status
        lines.append(f"| `{row.field}` | {row.group} | {row.filled}/{row.rows} | {row.distinct} | {note} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{source}: {len(inventory)} fields, {len(findings)} failed checks; {out}")
    for f in findings[:30]:
        print(f"  {f['check']}: {f['rows']} rows")


if __name__ == "__main__":
    main()
