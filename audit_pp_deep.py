"""Offline, price-aware audit of saved pregame NHL power-play point trackers.

The report is exploratory. Candidate fields were specified from the tracker
schema before reviewing the result table. Nothing here changes live rankings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_pp_early import named_main_move
from audit_pp_weights import auc


NUMERIC_FEATURES = (
    # Direct PP usage, history, chances, matchup.
    "PP_TOI_per_game", "PP_TOI_min", "PP_TOI_Share", "PP_TOI_Pct_Game",
    "PP_TeamShare_pct", "PP_TOI_stability", "PP_iXG60", "PP_iXA60",
    "PP_OnIce_xGF60", "PP_Points60", "PPP10_total", "Drought_PPP",
    "Team_PP_xGF60", "Team_PP_GF60", "Opp_PK_xGA60", "Opp_PK_GA60",
    "PP_Matchup", "PP_Env_Score",
    # Opponent defense and goalie.
    "Opp_GP", "Opp_GAA", "Opp_SV", "Goalie_Weak", "Opp_DefWeak",
    "opp_5v5_xGA60", "opp_5v5_HDCA60", "opp_5v5_SlotSA60",
    # Cross-market player opportunity and recent form.
    "TOI_per_game", "TOI_Pct", "L10_P", "L10_A", "L5_A",
    "Drought_P", "Drought_A", "Assist_Volume", "Assist_Dagger",
    "iXA%", "iXG%", "i5v5_primaryAssists60", "i5v5_shotAssists60",
    "i5v5_points60", "Goal_mu", "Assists_mu", "Points_mu",
    "Conf_Assists", "Conf_Points",
    "Team_GF_L5", "Team_GF_Avg_L5", "ShotIntent", "StarScore",
)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -25, 25)
    return 1 / (1 + np.exp(-value))


def _fit_offset(book: np.ndarray, y: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    """Fit a book-logit offset with an intercept and optional one-feature adjustment."""
    offset = np.log(book / (1 - book))
    design = np.ones((len(y), 1)) if x is None else np.column_stack((np.ones(len(y)), x))
    coef = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + ([2.0] if x is not None else []))
    for _ in range(40):
        probability = _sigmoid(offset + design @ coef)
        weight = np.maximum(probability * (1 - probability), 1e-6)
        gradient = design.T @ (y - probability) - penalty @ coef
        hessian = (design.T * weight) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        coef += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return coef


def _predict(book: np.ndarray, coefficient: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    offset = np.log(book / (1 - book))
    return _sigmoid(offset + coefficient[0] + (coefficient[1] * x if x is not None else 0))


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)), "dates": int(frame["Date"].nunique()),
        "hits": int(frame["hit"].sum()),
        "hit_pct": round(float(frame["hit"].mean() * 100), 1),
        "book_break_even_pct": round(float(frame["book"].mean() * 100), 1),
        "flat_roi_pct": round(float(frame["profit"].mean() * 100), 1),
    }


def _prepare(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames = []
    skipped = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {"Date", "Game", "Team", "Opp", "Player_ID", "Outcome_PPP",
                    "Grade_Status", "Match_Status_PPP", "BDL_PPP_Line", "BDL_PPP_Odds",
                    "Odds_Checked_UTC", "StartTimeUTC"}
        if not required.issubset(frame.columns):
            skipped.append({"file": path.name, "missing": sorted(required - set(frame.columns))})
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), {"files": len(paths), "skipped": skipped}
    all_rows = pd.concat(frames, ignore_index=True)
    line = pd.to_numeric(all_rows["BDL_PPP_Line"], errors="coerce")
    odds = pd.to_numeric(all_rows["BDL_PPP_Odds"], errors="coerce")
    price_rows = all_rows.loc[line.eq(0.5) & odds.notna() & odds.ne(0)].copy()
    checked = pd.to_datetime(price_rows["Odds_Checked_UTC"], errors="coerce", utc=True)
    started = pd.to_datetime(price_rows["StartTimeUTC"], errors="coerce", utc=True)
    price_rows["hours_before_start"] = (started - checked).dt.total_seconds() / 3600
    pregame = price_rows.loc[checked.notna() & started.notna() & checked.lt(started)].copy()
    eligible = pregame.loc[
        pregame["Grade_Status"].eq("FINAL")
        & pregame["Match_Status_PPP"].eq("OK")
        & pregame["Outcome_PPP"].isin(("W", "L"))
    ].copy()
    before_dedupe = len(eligible)
    eligible = eligible.drop_duplicates(["Date", "Game", "Team", "Player_ID"])
    odds = pd.to_numeric(eligible["BDL_PPP_Odds"])
    eligible["book"] = np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100))
    eligible["hit"] = eligible["Outcome_PPP"].eq("W").astype(int)
    eligible["profit"] = eligible["hit"] * np.where(odds > 0, odds / 100, 100 / -odds) - (1 - eligible["hit"])
    for field in NUMERIC_FEATURES:
        eligible[field] = pd.to_numeric(eligible.get(field), errors="coerce")
    for market in ("Assists", "Points"):
        eligible[f"{market}_Green"] = eligible.get(
            f"Matrix_{market}", pd.Series(index=eligible.index, dtype=object)
        ).astype(str).str.casefold().eq("green").astype(int)
    eligible["Opp_SV_weak"] = -eligible["Opp_SV"]
    role = eligible.get("PP_Role", pd.Series(index=eligible.index, dtype=float))
    eligible["PP1"] = pd.to_numeric(role, errors="coerce").ge(2).astype(int)
    eligible["PP_BOOST_ON"] = eligible.get("PP_BOOST", pd.Series(index=eligible.index, dtype=object)).eq("ON").astype(int)
    moves = eligible.get("Fired_Moves", pd.Series(index=eligible.index, dtype=object)).map(named_main_move)
    eligible["Named_Assist_Move"] = moves.map(lambda item: int(item[0]))
    eligible["Named_Points_Move"] = moves.map(lambda item: int(item[1]))
    eligible["PP1_x_weakPK"] = eligible["PP1"] * eligible["Opp_PK_xGA60"]
    eligible["PPtime_x_weakPK"] = eligible["PP_TOI_per_game"] * eligible["Opp_PK_xGA60"]
    eligible["PPtime_x_PPchance"] = eligible["PP_TOI_per_game"] * eligible["Team_PP_xGF60"]
    eligible["PPtime_x_goalie"] = eligible["PP_TOI_per_game"] * eligible["Goalie_Weak"]
    eligible["PPtime_x_assistmove"] = eligible["PP_TOI_per_game"] * eligible["Named_Assist_Move"]
    eligible["PPtime_x_pointsmove"] = eligible["PP_TOI_per_game"] * eligible["Named_Points_Move"]
    quality = {
        "files": len(paths), "skipped_files": skipped,
        "compatible_tracker_rows": int(len(all_rows)),
        "priced_over_0_5_rows": int(len(price_rows)),
        "priced_with_pregame_timestamp": int(len(pregame)),
        "pregame_final_graded_rows_before_dedupe": before_dedupe,
        "eligible_unique_props": int(len(eligible)),
        "pregame_ungraded_statuses": pregame.loc[~pregame.index.isin(eligible.index), "Grade_Status"].value_counts(dropna=False).to_dict(),
        "stats_source_counts": eligible.get("Model_Stats_Season", pd.Series(index=eligible.index, dtype=object)).value_counts(dropna=False).to_dict(),
        "quote_age_hours_median": round(float(eligible["hours_before_start"].median()), 1),
        "quotes_older_than_24h": int(eligible["hours_before_start"].gt(24).sum()),
        "best_price_book_counts": eligible.get(
            "BDL_PPP_Book", pd.Series(index=eligible.index, dtype=object)
        ).value_counts(dropna=False).head(12).to_dict(),
        "odds_source_counts": eligible.get(
            "PPP_Odds_Source", pd.Series(index=eligible.index, dtype=object)
        ).value_counts(dropna=False).to_dict(),
        "form_log_coverage": int(eligible.get("Form_Log", pd.Series(index=eligible.index, dtype=object)).notna().sum()),
    }
    return eligible, quality


def _scan_feature(frame: pd.DataFrame, name: str, early: set[str], late: set[str]) -> dict | None:
    pair = frame[["Date", "hit", "book", name]].dropna().copy()
    train = pair.loc[pair["Date"].isin(early)]
    test = pair.loc[pair["Date"].isin(late)]
    if (len(train) < 70 or len(test) < 70 or train["Date"].nunique() < 3
            or test["Date"].nunique() < 3 or train[name].std() < 1e-9):
        return None
    center, scale = float(train[name].mean()), float(train[name].std())
    x_train = ((train[name] - center) / scale).to_numpy(dtype=float)
    x_test = ((test[name] - center) / scale).to_numpy(dtype=float)
    y_train, y_test = train["hit"].to_numpy(dtype=float), test["hit"].to_numpy(dtype=float)
    b_train, b_test = train["book"].to_numpy(dtype=float), test["book"].to_numpy(dtype=float)
    baseline = _fit_offset(b_train, y_train)
    feature = _fit_offset(b_train, y_train, x_train)
    p_base = _predict(b_test, baseline)
    p_feature = _predict(b_test, feature, x_test)
    improvement = _log_loss(y_test, p_base) - _log_loss(y_test, p_feature)
    daily_gains = []
    daily_loss_sums = []
    daily_counts = []
    p_base_clipped = np.clip(p_base, 1e-6, 1 - 1e-6)
    p_feature_clipped = np.clip(p_feature, 1e-6, 1 - 1e-6)
    per_row_gain = (
        -y_test * np.log(p_base_clipped) - (1 - y_test) * np.log1p(-p_base_clipped)
        + y_test * np.log(p_feature_clipped) + (1 - y_test) * np.log1p(-p_feature_clipped)
    )
    for day in sorted(test["Date"].unique()):
        mask = test["Date"].eq(day).to_numpy()
        daily_gains.append(_log_loss(y_test[mask], p_base[mask]) -
                           _log_loss(y_test[mask], p_feature[mask]))
        daily_loss_sums.append(float(per_row_gain[mask].sum()))
        daily_counts.append(int(mask.sum()))
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    sample_indices = rng.integers(0, len(daily_counts), size=(2000, len(daily_counts)))
    bootstrapped = np.asarray(daily_loss_sums)[sample_indices].sum(axis=1) / (
        np.asarray(daily_counts)[sample_indices].sum(axis=1)
    )
    return {
        "field": name, "n_train": len(train), "n_test": len(test),
        "train_dates": int(train["Date"].nunique()), "test_dates": int(test["Date"].nunique()),
        "all_coverage_pct": round(100 * len(pair) / len(frame), 1),
        "train_coefficient": round(float(feature[1]), 3),
        "train_auc": auc(train["hit"], train[name]),
        "test_auc": auc(test["hit"], test[name]),
        "test_logloss_gain_vs_calibrated_book": round(improvement, 5),
        "test_brier_gain_vs_calibrated_book": round(
            float(np.mean((y_test - p_base) ** 2) - np.mean((y_test - p_feature) ** 2)), 5),
        "test_dates_helped": sum(gain > 0 for gain in daily_gains),
        "date_bootstrap_gain_95": [round(float(value), 5)
                                   for value in np.quantile(bootstrapped, [0.025, 0.975])],
    }


def build_report(paths: list[Path]) -> dict:
    frame, quality = _prepare(paths)
    if frame.empty:
        return {"status": "no_eligible_props", "quality": quality}
    dates = sorted(frame["Date"].astype(str).unique())
    cutoff = len(dates) // 2
    early, late = set(dates[:cutoff]), set(dates[cutoff:])
    features = list(NUMERIC_FEATURES) + [
        "Assists_Green", "Points_Green",
        "Opp_SV_weak", "PP1", "PP_BOOST_ON", "Named_Assist_Move",
        "Named_Points_Move", "PP1_x_weakPK", "PPtime_x_weakPK",
        "PPtime_x_PPchance", "PPtime_x_goalie", "PPtime_x_assistmove",
        "PPtime_x_pointsmove",
    ]
    tested = [result for name in features if (result := _scan_feature(frame, name, early, late))]
    tested.sort(key=lambda result: result["test_logloss_gain_vs_calibrated_book"], reverse=True)
    season_labels = frame.get("Model_Stats_Season", pd.Series(index=frame.index, dtype=object))
    latest_stats_season = max((label for label in season_labels.dropna().astype(str).unique()
                               if label[:4].isdigit()), default=None)
    return {
        "status": "exploratory_not_for_live_tuning",
        "quality": quality,
        "early_dates": dates[:cutoff], "later_dates": dates[cutoff:],
        "all": _summary(frame),
        "early": _summary(frame.loc[frame["Date"].isin(early)]),
        "later": _summary(frame.loc[frame["Date"].isin(late)]),
        "latest_stats_source": {
            "season": latest_stats_season,
            "summary": _summary(frame.loc[season_labels.eq(latest_stats_season)]),
        },
        "tested_fields": len(tested), "feature_scan": tested,
        "method": "Each field is tested alone as an adjustment to book implied probability. Early dates fit an intercept and one regularized slope; later dates are held out. A positive test log-loss gain means it improved probability estimates beyond recalibrating the book alone. Fields are correlated and many were screened; top-ranked gains are exploratory, not validated betting edges. Player props on the same game/team are correlated. No live model changes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = build_report(sorted((args.repo / "output" / "graded").glob("tracker_*_GRADED.csv")))
    destination = args.repo / "output" / "graded" / "pp_deep_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PP deep audit: {report['status']} -> {destination}")


if __name__ == "__main__":
    main()
