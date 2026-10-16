# Tracker column audit: tracker_latest.csv

Columns inventoried: **427**. This is a saved-snapshot arithmetic and completeness audit, not live-feed confirmation or model calibration.

## Findings

No tested cross-field invariant failed.

## Code and label defects found during this audit

- **Fixed for the next run:** 47 of 47 logged players had fewer than 10 completed games. The old L10/L20/L40 per-game rates divided by the window label instead of the observed number of games. The next tracker saves actual game counts.
- **Fixed for the next run:** displayed expectations and regression gaps disagreed for 21 assist rows and 15 points rows because a later multiplier changed the expectation. The next tracker exposes the exact expectation used for each gap.
- **Fixed for the next run:** the points matrix referenced an undefined confidence variable on one branch; a regression test now exercises that branch.
- **Fixed for the next run:** the old 5v5 SOG-share field used a partial slate proxy. The next tracker calculates player shots divided by all team skater shots in the 5v5 MoneyPuck feed; traded players' old-team shares stay blank.
- **Preserved transparently:** the historically graded Berserker Volley move continues to use its slate SOG share proxy, now named as such. The true 5v5 share is a separate field and has no inherited historical win rate.
- **Clarified for the next run:** PP iXA and PP points are observed-assist / mixed involvement proxies. Actual PP points/60 is a separate field; frozen historical move thresholds retain their old numeric inputs.
- **Fixed for the next run:** regression gaps now compare the 10-game expectation with a 10-game-equivalent observed pace; fewer than five recorded games leave the gap unavailable. A gap is still not a calibrated bounce-back probability.


## Known source and interpretation limits

- `iXA%` ranks observed season assists, because this MoneyPuck feed does not include xAssists. It is not an expected-assists percentile.
- `PP_iXA60` is weighted observed assists/60; `PP_Points60` mixes xG and observed assists. These legacy numbers are not expected assists or actual points.
- `PP_Role` is season-total PP-time rank, not a confirmed tonight unit. `PP_Matchup` and confidence fields are heuristics, not calibrated hit probabilities.
- Empty alternate-line columns mean those rungs were not quoted. Empty log fields may reflect the candidate-only NHL-log fetch.
- Recheck current prices and lineup/goalie status at the source before using a saved slate.

## Every field

| Field | Group | Filled | Distinct | Status / note |
|---|---|---:|---:|---|
| `Date` | identity/schedule | 82/82 | 1 | snapshot check passed |
| `Game` | identity/schedule | 82/82 | 2 | snapshot check passed |
| `Game_ID` | identity/schedule | 82/82 | 2 | snapshot check passed |
| `StartTimeLocal` | identity/schedule | 82/82 | 2 | snapshot check passed |
| `StartTimeUTC` | identity/schedule | 82/82 | 2 | snapshot check passed |
| `Player` | identity/schedule | 82/82 | 82 | snapshot check passed |
| `Player_ID` | identity/schedule | 82/82 | 82 | snapshot check passed |
| `Model_Stats_Team` | identity/schedule | 82/82 | 4 | snapshot check passed |
| `Team_Changed` | identity/schedule | 82/82 | 1 | snapshot check passed |
| `Form_Log` | official game-log feature | 47/82 | 47 | snapshot check passed |
| `Form_Season` | official game-log feature | 82/82 | 1 | snapshot check passed |
| `Team` | identity/schedule | 82/82 | 4 | snapshot check passed |
| `Pos` | identity/schedule | 82/82 | 4 | snapshot check passed |
| `Tier` | season stat/derived context | 4/82 | 2 | snapshot check passed |
| `Injury_Status` | availability/roster | 82/82 | 2 | snapshot check passed |
| `Injury_Type` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Injury_Text` | availability/roster | 0/82 | 0 | empty; empty in this slate |
| `Injury_Return` | availability/roster | 0/82 | 0 | empty; empty in this slate |
| `Team_Out_Count` | availability/roster | 82/82 | 2 | snapshot check passed |
| `Injury_DFO_Score` | availability/roster | 82/82 | 3 | snapshot check passed |
| `Injury_Badge` | availability/roster | 8/82 | 2 | snapshot check passed |
| `Injury_Reports_Count` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Injury_Excluded_Count` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Injury_Check_UTC` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Opp` | identity/schedule | 82/82 | 4 | snapshot check passed |
| `Opp_Goalie` | goalie context | 82/82 | 4 | snapshot check passed |
| `Opp_GP` | season stat/derived context | 82/82 | 3 | snapshot check passed |
| `Opp_SV` | goalie context | 82/82 | 4 | snapshot check passed |
| `Opp_GAA` | goalie context | 82/82 | 4 | snapshot check passed |
| `Goalie_Weak` | goalie context | 82/82 | 4 | snapshot check passed |
| `Opp_Goalie_Status` | goalie context | 82/82 | 1 | snapshot check passed |
| `Opp_Goalie_Source` | goalie context | 82/82 | 1 | snapshot check passed |
| `Opp_SOG_Against_L10` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Opp_SOG_Used_L10` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Opp_SOG_Against_L50` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Opp_SOG_Used_L50` | season stat/derived context | 82/82 | 2 | snapshot check passed |
| `Player_5v5_SOG_Share` | season stat/derived context | 47/82 | 18 | snapshot check passed |
| `iXG%` | season stat/derived context | 82/82 | 70 | snapshot check passed |
| `iXA%` | season stat/derived context | 82/82 | 8 | snapshot check passed |
| `iXA_Source` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `TOI_per_game` | season stat/derived context | 82/82 | 78 | snapshot check passed |
| `TOI_Pct` | season stat/derived context | 82/82 | 81 | snapshot check passed |
| `StarScore` | season stat/derived context | 82/82 | 76 | snapshot check passed |
| `TalentMult` | season stat/derived context | 82/82 | 24 | snapshot check passed |
| `Talent_Tier` | season stat/derived context | 82/82 | 3 | snapshot check passed |
| `Is_Elite` | season stat/derived context | 82/82 | 2 | snapshot check passed |
| `Is_Star` | season stat/derived context | 82/82 | 2 | snapshot check passed |
| `team_5v5_SF60` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `team_5v5_xGF60` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `team_5v5_SF60_pct` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `team_5v5_xGF60_pct` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Team_GF_L5` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Team_GF_Avg_L5` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Team_GF_Gate` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Med10_SOG` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `Trim10_SOG` | official game-log feature | 47/82 | 20 | snapshot check passed |
| `Avg5_SOG` | official game-log feature | 47/82 | 17 | snapshot check passed |
| `ShotIntent` | season stat/derived context | 47/82 | 25 | snapshot check passed |
| `ShotIntent_Pct` | season stat/derived context | 82/82 | 25 | snapshot check passed |
| `L10_P` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L10_G` | official game-log feature | 47/82 | 6 | snapshot check passed |
| `L10_S` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L5_G` | season stat/derived context | 47/82 | 4 | snapshot check passed |
| `L5_A` | season stat/derived context | 47/82 | 7 | snapshot check passed |
| `L10_A` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L20_P` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L40_P` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L20_A` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L40_A` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L20_S` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L40_S` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L10_Tier_Points` | official game-log feature | 47/82 | 4 | snapshot check passed |
| `L10_Rate_Points` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L10_Diff_Points` | official game-log feature | 34/82 | 10 | snapshot check passed |
| `L20_Tier_Points` | official game-log feature | 47/82 | 2 | snapshot check passed |
| `L20_Rate_Points` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L20_Diff_Points` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L40_Tier_Points` | official game-log feature | 47/82 | 1 | snapshot check passed |
| `L40_Rate_Points` | official game-log feature | 47/82 | 10 | snapshot check passed |
| `L40_Diff_Points` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L10_Tier_Assists` | official game-log feature | 47/82 | 3 | snapshot check passed |
| `L10_Rate_Assists` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L10_Diff_Assists` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L20_Tier_Assists` | official game-log feature | 47/82 | 1 | snapshot check passed |
| `L20_Rate_Assists` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L20_Diff_Assists` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L40_Tier_Assists` | official game-log feature | 47/82 | 1 | snapshot check passed |
| `L40_Rate_Assists` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `L40_Diff_Assists` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L10_Tier_SOG` | official game-log feature | 47/82 | 2 | snapshot check passed |
| `L10_Rate_SOG` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L10_Diff_SOG` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L20_Tier_SOG` | official game-log feature | 47/82 | 1 | snapshot check passed |
| `L20_Rate_SOG` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L20_Diff_SOG` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `L40_Tier_SOG` | official game-log feature | 47/82 | 1 | snapshot check passed |
| `L40_Rate_SOG` | official game-log feature | 47/82 | 19 | snapshot check passed |
| `L40_Diff_SOG` | official game-log feature | 0/82 | 0 | empty; empty in this slate |
| `i5v5_points60` | season stat/derived context | 82/82 | 69 | snapshot check passed |
| `i5v5_primaryAssists60` | season stat/derived context | 82/82 | 46 | snapshot check passed |
| `i5v5_shotAssists60` | season stat/derived context | 0/82 | 0 | empty; empty in this slate |
| `i5v5_iCF60` | season stat/derived context | 0/82 | 0 | empty; empty in this slate |
| `Assist_Volume` | decision heuristic | 82/82 | 8 | snapshot check passed |
| `opp_5v5_xGA60` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `opp_5v5_HDCA60` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `opp_5v5_SlotSA60` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Opp_DefWeak` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `v2_player_stability` | season stat/derived context | 82/82 | 69 | snapshot check passed |
| `v2_defense_vulnerability` | season stat/derived context | 82/82 | 4 | snapshot check passed |
| `Exp_P_10` | decision heuristic | 82/82 | 78 | snapshot check passed |
| `Exp_G_10` | decision heuristic | 82/82 | 75 | snapshot check passed |
| `Exp_S_10` | decision heuristic | 47/82 | 29 | snapshot check passed |
| `Reg_Gap_P10` | decision heuristic | 47/82 | 45 | snapshot check passed |
| `Reg_Gap_G10` | decision heuristic | 47/82 | 43 | snapshot check passed |
| `Reg_Gap_S10` | decision heuristic | 47/82 | 38 | snapshot check passed |
| `Reg_Heat_P` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Reg_Heat_G` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Reg_Heat_S` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Exp_A_10` | decision heuristic | 82/82 | 56 | snapshot check passed |
| `Reg_Gap_A10` | decision heuristic | 47/82 | 41 | snapshot check passed |
| `Reg_Heat_A` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Matrix_Points` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Matrix_SOG` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Matrix_Goal` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Matrix_Assists` | decision heuristic | 82/82 | 3 | snapshot check passed |
| `Conf_Points` | decision heuristic | 82/82 | 55 | snapshot check passed |
| `Conf_SOG` | decision heuristic | 82/82 | 47 | snapshot check passed |
| `Conf_Goal` | decision heuristic | 82/82 | 52 | snapshot check passed |
| `Conf_Assists` | decision heuristic | 82/82 | 48 | snapshot check passed |
| `Best_Market` | decision heuristic | 82/82 | 4 | snapshot check passed |
| `Best_Conf` | decision heuristic | 82/82 | 46 | snapshot check passed |
| `Drought_P` | official game-log feature | 47/82 | 5 | snapshot check passed |
| `Drought_A` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `Drought_G` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `Drought_SOG` | official game-log feature | 47/82 | 8 | snapshot check passed |
| `PPP10_total` | official game-log feature | 47/82 | 6 | snapshot check passed |
| `Drought_PPP` | official game-log feature | 47/82 | 7 | snapshot check passed |
| `PP_Tier` | power-play data/heuristic | 82/82 | 3 | snapshot check passed |
| `PP_Path` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `PP_TeamShare_pct` | power-play data/heuristic | 82/82 | 57 | snapshot check passed |
| `PP_TOI_stability` | power-play data/heuristic | 0/82 | 0 | empty; empty in this slate |
| `PP_Env_Score` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `PP_BOOST` | power-play data/heuristic | 1/82 | 1 | snapshot check passed |
| `PP_Role` | power-play data/heuristic | 82/82 | 3 | snapshot check passed |
| `PP_TOI` | power-play data/heuristic | 82/82 | 70 | snapshot check passed |
| `PP_TOI_min` | power-play data/heuristic | 82/82 | 66 | snapshot check passed |
| `PP_TOI_per_game` | power-play data/heuristic | 82/82 | 70 | snapshot check passed |
| `PP_TOI_Share` | power-play data/heuristic | 82/82 | 74 | snapshot check passed |
| `PP_TOI_Pct_Game` | power-play data/heuristic | 82/82 | 57 | snapshot check passed |
| `PP_TOI_Pct` | power-play data/heuristic | 82/82 | 57 | snapshot check passed |
| `PP_Points60` | power-play data/heuristic | 82/82 | 38 | snapshot check passed |
| `PP_iXG60` | power-play data/heuristic | 73/82 | 34 | snapshot check passed |
| `PP_OnIce_xGF60` | power-play data/heuristic | 73/82 | 45 | snapshot check passed |
| `PP_iXA60` | power-play data/heuristic | 73/82 | 20 | snapshot check passed |
| `Team_PP_xGF60` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `Team_PP_GF60` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `Opp_PK_xGA60` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `Opp_PK_GA60` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `PP_Matchup` | power-play data/heuristic | 82/82 | 4 | snapshot check passed |
| `Assist_Dagger` | decision heuristic | 82/82 | 64 | snapshot check passed |
| `Assist_PP_Proof` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Line` | empty grading placeholder | 0/82 | 0 | empty; empty in this slate |
| `Odds` | empty grading placeholder | 0/82 | 0 | empty; empty in this slate |
| `Result` | empty grading placeholder | 0/82 | 0 | empty; empty in this slate |
| `Best_Drought` | decision heuristic | 47/82 | 17 | snapshot check passed |
| `Play_Tag` | season stat/derived context | 1/82 | 1 | snapshot check passed |
| `Plays_Points` | decision heuristic | 82/82 | 1 | snapshot check passed |
| `Plays_Goal` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Goal_ProofCount` | decision heuristic | 82/82 | 4 | snapshot check passed |
| `Goal_Why` | decision heuristic | 1/82 | 1 | snapshot check passed |
| `Plays_Assists` | decision heuristic | 82/82 | 1 | snapshot check passed |
| `Assist_ProofCount` | decision heuristic | 82/82 | 1 | snapshot check passed |
| `Assist_Why` | decision heuristic | 0/82 | 0 | empty; empty in this slate |
| `Plays_SOG` | decision heuristic | 82/82 | 1 | snapshot check passed |
| `SOG_ProofCount` | decision heuristic | 82/82 | 1 | snapshot check passed |
| `SOG_Why` | decision heuristic | 0/82 | 0 | empty; empty in this slate |
| `BDL_SOG_Line_1` | captured book quote | 44/82 | 2 | snapshot check passed |
| `BDL_SOG_Odds_1` | captured book quote | 44/82 | 36 | snapshot check passed |
| `BDL_SOG_Book_1` | captured book quote | 44/82 | 9 | snapshot check passed |
| `BDL_SOG_Line_2` | captured book quote | 36/82 | 1 | snapshot check passed |
| `BDL_SOG_Odds_2` | captured book quote | 36/82 | 30 | snapshot check passed |
| `BDL_SOG_Book_2` | captured book quote | 36/82 | 7 | snapshot check passed |
| `BDL_SOG_Line_3` | captured book quote | 36/82 | 1 | snapshot check passed |
| `BDL_SOG_Odds_3` | captured book quote | 36/82 | 23 | snapshot check passed |
| `BDL_SOG_Book_3` | captured book quote | 36/82 | 6 | snapshot check passed |
| `BDL_SOG_Line_4` | captured book quote | 36/82 | 1 | snapshot check passed |
| `BDL_SOG_Odds_4` | captured book quote | 36/82 | 24 | snapshot check passed |
| `BDL_SOG_Book_4` | captured book quote | 36/82 | 4 | snapshot check passed |
| `BDL_SOG_Line` | captured book quote | 44/82 | 3 | snapshot check passed |
| `BDL_SOG_Odds` | captured book quote | 44/82 | 30 | snapshot check passed |
| `BDL_SOG_Book` | captured book quote | 44/82 | 8 | snapshot check passed |
| `BDL_Points_Line_1` | captured book quote | 46/82 | 1 | snapshot check passed |
| `BDL_Points_Odds_1` | captured book quote | 46/82 | 35 | snapshot check passed |
| `BDL_Points_Book_1` | captured book quote | 46/82 | 8 | snapshot check passed |
| `BDL_Points_Line_2` | captured book quote | 37/82 | 1 | snapshot check passed |
| `BDL_Points_Odds_2` | captured book quote | 37/82 | 25 | snapshot check passed |
| `BDL_Points_Book_2` | captured book quote | 37/82 | 5 | snapshot check passed |
| `BDL_Points_Line_3` | captured book quote | 32/82 | 1 | snapshot check passed |
| `BDL_Points_Odds_3` | captured book quote | 32/82 | 18 | snapshot check passed |
| `BDL_Points_Book_3` | captured book quote | 32/82 | 4 | snapshot check passed |
| `BDL_Points_Line_4` | captured book quote | 32/82 | 1 | snapshot check passed |
| `BDL_Points_Odds_4` | captured book quote | 32/82 | 18 | snapshot check passed |
| `BDL_Points_Book_4` | captured book quote | 32/82 | 2 | snapshot check passed |
| `BDL_Points_Line` | captured book quote | 46/82 | 2 | snapshot check passed |
| `BDL_Points_Odds` | captured book quote | 46/82 | 34 | snapshot check passed |
| `BDL_Points_Book` | captured book quote | 46/82 | 8 | snapshot check passed |
| `BDL_Assists_Line_1` | captured book quote | 42/82 | 1 | snapshot check passed |
| `BDL_Assists_Odds_1` | captured book quote | 42/82 | 28 | snapshot check passed |
| `BDL_Assists_Book_1` | captured book quote | 42/82 | 9 | snapshot check passed |
| `BDL_Assists_Line_2` | captured book quote | 32/82 | 1 | snapshot check passed |
| `BDL_Assists_Odds_2` | captured book quote | 32/82 | 15 | snapshot check passed |
| `BDL_Assists_Book_2` | captured book quote | 32/82 | 3 | snapshot check passed |
| `BDL_Assists_Line_3` | captured book quote | 14/82 | 1 | snapshot check passed |
| `BDL_Assists_Odds_3` | captured book quote | 14/82 | 9 | snapshot check passed |
| `BDL_Assists_Book_3` | captured book quote | 14/82 | 2 | snapshot check passed |
| `BDL_Assists_Line_4` | captured book quote | 2/82 | 1 | snapshot check passed |
| `BDL_Assists_Odds_4` | captured book quote | 2/82 | 2 | snapshot check passed |
| `BDL_Assists_Book_4` | captured book quote | 2/82 | 1 | snapshot check passed |
| `BDL_Assists_Line` | captured book quote | 42/82 | 1 | snapshot check passed |
| `BDL_Assists_Odds` | captured book quote | 42/82 | 28 | snapshot check passed |
| `BDL_Assists_Book` | captured book quote | 42/82 | 9 | snapshot check passed |
| `BDL_Goal_Line_1` | captured book quote | 70/82 | 1 | snapshot check passed |
| `BDL_Goal_Odds_1` | captured book quote | 70/82 | 38 | snapshot check passed |
| `BDL_Goal_Book_1` | captured book quote | 70/82 | 7 | snapshot check passed |
| `BDL_Goal_Line_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Line_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Line_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_Goal_Line` | captured book quote | 70/82 | 1 | snapshot check passed |
| `BDL_Goal_Odds` | captured book quote | 70/82 | 38 | snapshot check passed |
| `BDL_Goal_Book` | captured book quote | 70/82 | 7 | snapshot check passed |
| `BDL_ATG_Line_1` | captured book quote | 71/82 | 1 | snapshot check passed |
| `BDL_ATG_Odds_1` | captured book quote | 71/82 | 40 | snapshot check passed |
| `BDL_ATG_Book_1` | captured book quote | 71/82 | 6 | snapshot check passed |
| `BDL_ATG_Line_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_2` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Line_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Line_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_ATG_Line` | captured book quote | 71/82 | 1 | snapshot check passed |
| `BDL_ATG_Odds` | captured book quote | 71/82 | 40 | snapshot check passed |
| `BDL_ATG_Book` | captured book quote | 71/82 | 6 | snapshot check passed |
| `PPP_Book_Prices_05` | season stat/derived context | 82/82 | 35 | snapshot check passed |
| `Goal_Odds_Source` | captured book quote | 70/82 | 1 | snapshot check passed |
| `Goal_Available_Books` | captured book quote | 70/82 | 5 | snapshot check passed |
| `SOG_Odds_Source` | captured book quote | 44/82 | 2 | snapshot check passed |
| `SOG_Available_Books` | captured book quote | 44/82 | 12 | snapshot check passed |
| `ATG_Odds_Source` | captured book quote | 71/82 | 2 | snapshot check passed |
| `ATG_Available_Books` | captured book quote | 71/82 | 9 | snapshot check passed |
| `Assists_Odds_Source` | captured book quote | 42/82 | 2 | snapshot check passed |
| `Assists_Available_Books` | captured book quote | 42/82 | 8 | snapshot check passed |
| `BDL_PPP_Line_1` | captured book quote | 38/82 | 1 | snapshot check passed |
| `BDL_PPP_Odds_1` | captured book quote | 38/82 | 25 | snapshot check passed |
| `BDL_PPP_Book_1` | captured book quote | 38/82 | 3 | snapshot check passed |
| `BDL_PPP_Line_2` | captured book quote | 13/82 | 1 | snapshot check passed |
| `BDL_PPP_Odds_2` | captured book quote | 13/82 | 8 | snapshot check passed |
| `BDL_PPP_Book_2` | captured book quote | 13/82 | 1 | snapshot check passed |
| `BDL_PPP_Line_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Odds_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Book_3` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Line_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Odds_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Book_4` | captured book quote | 0/82 | 0 | empty; empty in this slate |
| `BDL_PPP_Line` | captured book quote | 38/82 | 1 | snapshot check passed |
| `BDL_PPP_Odds` | captured book quote | 38/82 | 25 | snapshot check passed |
| `BDL_PPP_Book` | captured book quote | 38/82 | 3 | snapshot check passed |
| `PPP_Odds_Source` | captured book quote | 38/82 | 1 | snapshot check passed |
| `PPP_Available_Books` | captured book quote | 38/82 | 7 | snapshot check passed |
| `Points_Odds_Source` | captured book quote | 46/82 | 2 | snapshot check passed |
| `Points_Available_Books` | captured book quote | 46/82 | 8 | snapshot check passed |
| `Odds_Name_Ambiguous` | captured book quote | 82/82 | 1 | snapshot check passed |
| `Odds_Checked_UTC` | captured book quote | 82/82 | 1 | snapshot check passed |
| `BDL_EV_ENGINE` | captured book quote | 82/82 | 1 | snapshot check passed |
| `SOG_Line` | selected book quote | 44/82 | 3 | snapshot check passed |
| `SOG_Odds_Over` | selected book quote | 44/82 | 30 | snapshot check passed |
| `SOG_Book` | selected book quote | 44/82 | 8 | snapshot check passed |
| `SOG_ModelSrc` | model/price calculation | 82/82 | 2 | snapshot check passed |
| `SOG_mu` | model/price calculation | 82/82 | 30 | snapshot check passed |
| `SOG_p_model_over` | model/price calculation | 44/82 | 31 | snapshot check passed |
| `SOG_Model%` | model/price calculation | 44/82 | 31 | snapshot check passed |
| `SOG_p_imp_over` | model/price calculation | 44/82 | 30 | snapshot check passed |
| `SOG_EVpct_over` | model/price calculation | 44/82 | 44 | snapshot check passed |
| `SOG_EV%` | model/price calculation | 44/82 | 44 | snapshot check passed |
| `SOG_p_model_over_1` | model/price calculation | 44/82 | 27 | snapshot check passed |
| `SOG_Model%_1` | model/price calculation | 44/82 | 27 | snapshot check passed |
| `SOG_EVpct_over_1` | model/price calculation | 44/82 | 44 | snapshot check passed |
| `SOG_EV%_1` | model/price calculation | 44/82 | 44 | snapshot check passed |
| `SOG_p_model_over_2` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_Model%_2` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_EVpct_over_2` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `SOG_EV%_2` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `SOG_p_model_over_3` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_Model%_3` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_EVpct_over_3` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `SOG_EV%_3` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `SOG_p_model_over_4` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_Model%_4` | model/price calculation | 36/82 | 24 | snapshot check passed |
| `SOG_EVpct_over_4` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `SOG_EV%_4` | model/price calculation | 36/82 | 36 | snapshot check passed |
| `Points_Line` | selected book quote | 46/82 | 2 | snapshot check passed |
| `Points_Odds_Over` | selected book quote | 46/82 | 34 | snapshot check passed |
| `Points_Book` | selected book quote | 46/82 | 8 | snapshot check passed |
| `Points_ModelSrc` | model/price calculation | 82/82 | 2 | snapshot check passed |
| `Points_mu` | model/price calculation | 82/82 | 78 | snapshot check passed |
| `Points_p_model_over` | model/price calculation | 46/82 | 44 | snapshot check passed |
| `Points_Model%` | model/price calculation | 46/82 | 43 | snapshot check passed |
| `Points_p_imp_over` | model/price calculation | 46/82 | 34 | snapshot check passed |
| `Points_EVpct_over` | model/price calculation | 46/82 | 46 | snapshot check passed |
| `Points_EV%` | model/price calculation | 46/82 | 44 | snapshot check passed |
| `Points_p_model_over_1` | model/price calculation | 46/82 | 44 | snapshot check passed |
| `Points_Model%_1` | model/price calculation | 46/82 | 44 | snapshot check passed |
| `Points_EVpct_over_1` | model/price calculation | 46/82 | 46 | snapshot check passed |
| `Points_EV%_1` | model/price calculation | 46/82 | 44 | snapshot check passed |
| `Points_p_model_over_2` | model/price calculation | 37/82 | 36 | snapshot check passed |
| `Points_Model%_2` | model/price calculation | 37/82 | 36 | snapshot check passed |
| `Points_EVpct_over_2` | model/price calculation | 37/82 | 37 | snapshot check passed |
| `Points_EV%_2` | model/price calculation | 37/82 | 36 | snapshot check passed |
| `Points_p_model_over_3` | model/price calculation | 32/82 | 31 | snapshot check passed |
| `Points_Model%_3` | model/price calculation | 32/82 | 29 | snapshot check passed |
| `Points_EVpct_over_3` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Points_EV%_3` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Points_p_model_over_4` | model/price calculation | 32/82 | 31 | snapshot check passed |
| `Points_Model%_4` | model/price calculation | 32/82 | 25 | snapshot check passed |
| `Points_EVpct_over_4` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Points_EV%_4` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Assists_Line` | selected book quote | 42/82 | 1 | snapshot check passed |
| `Assists_Odds_Over` | selected book quote | 42/82 | 28 | snapshot check passed |
| `Assists_Book` | selected book quote | 42/82 | 9 | snapshot check passed |
| `Assists_ModelSrc` | model/price calculation | 82/82 | 2 | snapshot check passed |
| `Assists_mu` | model/price calculation | 82/82 | 56 | snapshot check passed |
| `Assists_p_model_over` | model/price calculation | 42/82 | 36 | snapshot check passed |
| `Assists_Model%` | model/price calculation | 42/82 | 35 | snapshot check passed |
| `Assists_p_imp_over` | model/price calculation | 42/82 | 28 | snapshot check passed |
| `Assists_EVpct_over` | model/price calculation | 42/82 | 42 | snapshot check passed |
| `Assists_EV%` | model/price calculation | 42/82 | 42 | snapshot check passed |
| `Assists_p_model_over_1` | model/price calculation | 42/82 | 36 | snapshot check passed |
| `Assists_Model%_1` | model/price calculation | 42/82 | 35 | snapshot check passed |
| `Assists_EVpct_over_1` | model/price calculation | 42/82 | 42 | snapshot check passed |
| `Assists_EV%_1` | model/price calculation | 42/82 | 42 | snapshot check passed |
| `Assists_p_model_over_2` | model/price calculation | 32/82 | 28 | snapshot check passed |
| `Assists_Model%_2` | model/price calculation | 32/82 | 25 | snapshot check passed |
| `Assists_EVpct_over_2` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Assists_EV%_2` | model/price calculation | 32/82 | 32 | snapshot check passed |
| `Assists_p_model_over_3` | model/price calculation | 14/82 | 13 | snapshot check passed |
| `Assists_Model%_3` | model/price calculation | 14/82 | 13 | snapshot check passed |
| `Assists_EVpct_over_3` | model/price calculation | 14/82 | 14 | snapshot check passed |
| `Assists_EV%_3` | model/price calculation | 14/82 | 14 | snapshot check passed |
| `Assists_p_model_over_4` | model/price calculation | 2/82 | 2 | snapshot check passed |
| `Assists_Model%_4` | model/price calculation | 2/82 | 2 | snapshot check passed |
| `Assists_EVpct_over_4` | model/price calculation | 2/82 | 2 | snapshot check passed |
| `Assists_EV%_4` | model/price calculation | 2/82 | 2 | snapshot check passed |
| `Goal_Line` | selected book quote | 71/82 | 1 | snapshot check passed |
| `Goal_Odds_Over` | selected book quote | 71/82 | 39 | snapshot check passed |
| `Goal_Book` | selected book quote | 71/82 | 8 | snapshot check passed |
| `Goal_ModelSrc` | model/price calculation | 82/82 | 2 | snapshot check passed |
| `Goal_mu` | model/price calculation | 82/82 | 79 | snapshot check passed |
| `Goal_p_model_over` | model/price calculation | 70/82 | 67 | snapshot check passed |
| `Goal_Model%` | model/price calculation | 70/82 | 55 | snapshot check passed |
| `Goal_p_imp_over` | model/price calculation | 70/82 | 38 | snapshot check passed |
| `Goal_EVpct_over` | model/price calculation | 70/82 | 70 | snapshot check passed |
| `Goal_EV%` | model/price calculation | 70/82 | 68 | snapshot check passed |
| `Goal_p_model_over_1` | model/price calculation | 70/82 | 67 | snapshot check passed |
| `Goal_Model%_1` | model/price calculation | 70/82 | 55 | snapshot check passed |
| `Goal_EVpct_over_1` | model/price calculation | 70/82 | 70 | snapshot check passed |
| `Goal_EV%_1` | model/price calculation | 70/82 | 68 | snapshot check passed |
| `Goal_p_model_over_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_Model%_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EV%_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_p_model_over_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_Model%_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EV%_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_p_model_over_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_Model%_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Goal_EV%_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_Line` | selected book quote | 71/82 | 1 | snapshot check passed |
| `ATG_Odds_Over` | selected book quote | 71/82 | 40 | snapshot check passed |
| `ATG_Book` | selected book quote | 71/82 | 6 | snapshot check passed |
| `ATG_ModelSrc` | model/price calculation | 82/82 | 2 | snapshot check passed |
| `ATG_mu` | model/price calculation | 82/82 | 80 | snapshot check passed |
| `ATG_p_model_over` | model/price calculation | 71/82 | 69 | snapshot check passed |
| `ATG_Model%` | model/price calculation | 71/82 | 53 | snapshot check passed |
| `ATG_p_imp_over` | model/price calculation | 71/82 | 40 | snapshot check passed |
| `ATG_EVpct_over` | model/price calculation | 71/82 | 71 | snapshot check passed |
| `ATG_EV%` | model/price calculation | 71/82 | 69 | snapshot check passed |
| `ATG_p_model_over_1` | model/price calculation | 71/82 | 69 | snapshot check passed |
| `ATG_Model%_1` | model/price calculation | 71/82 | 53 | snapshot check passed |
| `ATG_EVpct_over_1` | model/price calculation | 71/82 | 71 | snapshot check passed |
| `ATG_EV%_1` | model/price calculation | 71/82 | 69 | snapshot check passed |
| `ATG_p_model_over_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_Model%_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EV%_2` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_p_model_over_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_Model%_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EV%_3` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_p_model_over_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_Model%_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `ATG_EV%_4` | model/price calculation | 0/82 | 0 | empty; empty in this slate |
| `Plays_EV_SOG` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Plays_EV_Points` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Plays_EV_Goal` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Plays_EV_ATG` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Plays_EV_Assists` | decision heuristic | 82/82 | 2 | snapshot check passed |
| `Player_SOG_Share_Proxy` | season stat/derived context | 47/82 | 18 | snapshot check passed |
| `Outcome` | empty grading placeholder | 0/82 | 0 | empty; empty in this slate |
| `Roster_Status` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Roster_Check_UTC` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Roster_Watch` | availability/roster | 82/82 | 1 | snapshot check passed |
| `Move_Kit_Version` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Fired_Moves` | decision heuristic | 82/82 | 5 | snapshot check passed |
| `Model_Stats_Season` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Team_Stats_Season` | season stat/derived context | 82/82 | 1 | snapshot check passed |
| `Goalie_Stats_Season` | goalie context | 82/82 | 1 | snapshot check passed |
