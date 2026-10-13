"""Track provisional PP matchup weights against frozen, graded current-season slates.

This is an observational ranking audit. It does not tune or change the live board.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


WEIGHTS = (0.0, 0.25, 0.45, 0.55, 0.75, 1.0)
REQUIRED = {
    "Date", "Game", "Team", "Opp", "Player_ID", "Grade_Status",
    "Match_Status_PPP", "BDL_PPP_Line", "Outcome_PPP",
    "Odds_Checked_UTC", "StartTimeUTC", "Team_PP_xGF60", "Opp_PK_xGA60",
    "PP_Matchup", "PP_Role",
}


def auc(outcomes: pd.Series, scores: pd.Series) -> float | None:
    paired = pd.DataFrame({"hit": outcomes, "score": scores}).dropna()
    hits = int(paired["hit"].sum())
    misses = len(paired) - hits
    if hits == 0 or misses == 0:
        return None
    ranks = paired["score"].rank(method="average")
    return round(float((ranks[paired["hit"].eq(1)].sum() - hits * (hits + 1) / 2) / (hits * misses)), 4)


def subset_result(frame: pd.DataFrame, dates: list[str]) -> dict:
    midpoint = len(dates) // 2
    early = frame["Date"].isin(dates[:midpoint])
    weights = []
    for offense_weight in WEIGHTS:
        score = offense_weight * frame["team_pct"] + (1 - offense_weight) * frame["opp_pct"]
        weights.append({
            "team_pp_weight": offense_weight,
            "opponent_pk_weight": round(1 - offense_weight, 2),
            "auc": auc(frame["hit"], score),
            "early_auc": auc(frame.loc[early, "hit"], score.loc[early]),
            "later_auc": auc(frame.loc[~early, "hit"], score.loc[~early]),
        })
    return {
        "player_props": len(frame),
        "hits": int(frame["hit"].sum()),
        "dates": int(frame["Date"].nunique()),
        "saved_live_55_45_auc": auc(frame["hit"], frame["PP_Matchup"]),
        "weights": weights,
    }


def build_report(paths: list[Path]) -> dict:
    if not paths:
        return {"status": "no_graded_slates", "candidate_weights": list(WEIGHTS)}
    rows = []
    skipped_files = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = REQUIRED.difference(frame.columns)
        if missing:
            skipped_files.append({"file": path.name, "missing_columns": sorted(missing)})
            continue
        rows.append(frame)
    if not rows:
        return {"status": "no_compatible_graded_slates", "skipped_files": skipped_files}
    all_rows = pd.concat(rows, ignore_index=True)
    for column in ("Team_PP_xGF60", "Opp_PK_xGA60", "PP_Matchup", "BDL_PPP_Line", "PP_Role"):
        all_rows[column] = pd.to_numeric(all_rows[column], errors="coerce")

    # Match each historical slate's available-team context, before filtering to priced props.
    team_pp = all_rows.groupby(["Date", "Team"])["Team_PP_xGF60"].median()
    team_pct = team_pp.groupby(level=0).rank(pct=True).rename("team_pct")
    opp_pk = all_rows.groupby(["Date", "Opp"])["Opp_PK_xGA60"].median()
    opp_pct = opp_pk.groupby(level=0).rank(pct=True).rename("opp_pct")

    graded = all_rows.loc[
        all_rows["Grade_Status"].eq("FINAL")
        & all_rows["Match_Status_PPP"].eq("OK")
        & all_rows["BDL_PPP_Line"].eq(0.5)
        & all_rows["Outcome_PPP"].isin(("W", "L"))
    ].copy()
    checked = pd.to_datetime(graded["Odds_Checked_UTC"], errors="coerce", utc=True)
    started = pd.to_datetime(graded["StartTimeUTC"], errors="coerce", utc=True)
    graded = graded.loc[checked.notna() & started.notna() & checked.lt(started)].copy()
    graded = graded.drop_duplicates(["Date", "Game", "Team", "Player_ID"])
    graded = graded.join(team_pct, on=["Date", "Team"]).join(opp_pct, on=["Date", "Opp"])
    graded = graded.dropna(subset=["team_pct", "opp_pct", "PP_Matchup"])
    if graded.empty:
        return {"status": "no_eligible_pregame_pp_props", "graded_files": len(paths)}
    graded["hit"] = graded["Outcome_PPP"].eq("W").astype(int)
    dates = sorted(graded["Date"].unique().tolist())
    return {
        "status": "exploratory",
        "first_date": dates[0],
        "last_date": dates[-1],
        "early_dates": dates[:len(dates) // 2],
        "later_dates": dates[len(dates) // 2:],
        "graded_files": len(paths),
        "skipped_files": skipped_files,
        "all_priced": subset_result(graded, dates),
        "inferred_pp1": subset_result(graded[graded["PP_Role"].ge(2)], dates),
        "method_note": "Candidate weights use within-slate team percentiles; the saved live score used league percentiles. AUC is descriptive, not a betting edge. Inferred PP1 is based on saved usage, not a confirmed game-day unit. No live rating is changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted((args.repo / "output" / "graded").glob("tracker_*_GRADED.csv"))
    report = build_report(paths)
    destination = args.output or args.repo / "output" / "graded" / "pp_weight_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(destination)
    print(f"PP weight audit: {report['status']} -> {destination}")


if __name__ == "__main__":
    main()
