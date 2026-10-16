# Tracker column audit: tracker_latest.csv

Columns inventoried: **421**. This is a saved-snapshot arithmetic and completeness audit, not live-feed confirmation or model calibration.

## Findings

No tested cross-field invariant failed.

## Code and label defects found during this audit

- **Fixed for the next run:** 178 of 178 logged players had fewer than 10 completed games. The old L10/L20/L40 per-game rates divided by the window label instead of the observed number of games. The next tracker saves actual game counts.
- **Fixed for the next run:** displayed expectations and regression gaps disagreed for 80 assist rows and 57 points rows because a later multiplier changed the expectation. The next tracker exposes the exact expectation used for each gap.
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
| `Date` | identity/schedule | 439/439 | 1 | snapshot check passed |
| `Game` | identity/schedule | 439/439 | 11 | snapshot check passed |
| `Game_ID` | identity/schedule | 439/439 | 11 | snapshot check passed |
| `StartTimeLocal` | identity/schedule | 439/439 | 5 | snapshot check passed |
| `StartTimeUTC` | identity/schedule | 439/439 | 5 | snapshot check passed |
| `Player` | identity/schedule | 439/439 | 438 | snapshot check passed |
| `Player_ID` | identity/schedule | 439/439 | 439 | snapshot check passed |
| `Model_Stats_Team` | identity/schedule | 436/439 | 22 | snapshot check passed |
| `Team_Changed` | identity/schedule | 436/439 | 1 | snapshot check passed |
| `Form_Log` | official game-log feature | 178/439 | 178 | snapshot check passed |
| `Form_Season` | official game-log feature | 436/439 | 1 | snapshot check passed |
| `Team` | identity/schedule | 439/439 | 22 | snapshot check passed |
| `Pos` | identity/schedule | 439/439 | 4 | snapshot check passed |
| `Tier` | season stat/derived context | 20/439 | 2 | snapshot check passed |
| `Injury_Status` | availability/roster | 439/439 | 3 | snapshot check passed |
| `Injury_Type` | availability/roster | 436/439 | 1 | snapshot check passed |
| `Injury_Text` | availability/roster | 0/439 | 0 | empty; empty in this slate |
| `Injury_Return` | availability/roster | 0/439 | 0 | empty; empty in this slate |
| `Team_Out_Count` | availability/roster | 436/439 | 6 | snapshot check passed |
| `Injury_DFO_Score` | availability/roster | 436/439 | 3 | snapshot check passed |
| `Injury_Badge` | availability/roster | 107/439 | 2 | snapshot check passed |
| `Injury_Reports_Count` | availability/roster | 436/439 | 1 | snapshot check passed |
| `Injury_Excluded_Count` | availability/roster | 436/439 | 1 | snapshot check passed |
| `Injury_Check_UTC` | availability/roster | 436/439 | 1 | snapshot check passed |
| `Opp` | identity/schedule | 439/439 | 22 | snapshot check passed |
| `Opp_Goalie` | goalie context | 436/439 | 22 | snapshot check passed |
| `Opp_SV` | goalie context | 417/439 | 21 | snapshot check passed |
| `Opp_GAA` | goalie context | 417/439 | 21 | snapshot check passed |
| `Goalie_Weak` | goalie context | 436/439 | 16 | snapshot check passed |
| `Opp_Goalie_Status` | goalie context | 436/439 | 3 | snapshot check passed |
| `Opp_Goalie_Source` | goalie context | 436/439 | 2 | snapshot check passed |
| `Opp_SOG_Against_L10` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `Opp_SOG_Used_L10` | season stat/derived context | 436/439 | 1 | snapshot check passed |
| `Opp_SOG_Against_L50` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `Opp_SOG_Used_L50` | season stat/derived context | 436/439 | 3 | snapshot check passed |
| `Player_5v5_SOG_Share` | season stat/derived context | 178/439 | 62 | snapshot check passed |
| `iXG%` | season stat/derived context | 436/439 | 205 | snapshot check passed |
| `iXA%` | season stat/derived context | 436/439 | 11 | snapshot check passed |
| `iXA_Source` | season stat/derived context | 436/439 | 1 | snapshot check passed |
| `TOI_per_game` | season stat/derived context | 436/439 | 381 | snapshot check passed |
| `TOI_Pct` | season stat/derived context | 436/439 | 429 | snapshot check passed |
| `StarScore` | season stat/derived context | 436/439 | 343 | snapshot check passed |
| `TalentMult` | season stat/derived context | 436/439 | 31 | snapshot check passed |
| `Talent_Tier` | season stat/derived context | 436/439 | 3 | snapshot check passed |
| `Is_Elite` | season stat/derived context | 436/439 | 2 | snapshot check passed |
| `Is_Star` | season stat/derived context | 436/439 | 2 | snapshot check passed |
| `team_5v5_SF60` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `team_5v5_xGF60` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `team_5v5_SF60_pct` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `team_5v5_xGF60_pct` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `Team_GF_L5` | season stat/derived context | 436/439 | 14 | snapshot check passed |
| `Team_GF_Avg_L5` | season stat/derived context | 436/439 | 14 | snapshot check passed |
| `Team_GF_Gate` | season stat/derived context | 436/439 | 1 | snapshot check passed |
| `Med10_SOG` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `Trim10_SOG` | official game-log feature | 175/439 | 41 | snapshot check passed |
| `Avg5_SOG` | official game-log feature | 178/439 | 25 | snapshot check passed |
| `ShotIntent` | season stat/derived context | 178/439 | 64 | snapshot check passed |
| `ShotIntent_Pct` | season stat/derived context | 436/439 | 65 | snapshot check passed |
| `L10_P` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L10_G` | official game-log feature | 178/439 | 9 | snapshot check passed |
| `L10_S` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L5_G` | season stat/derived context | 178/439 | 8 | snapshot check passed |
| `L5_A` | season stat/derived context | 178/439 | 9 | snapshot check passed |
| `L10_A` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L20_P` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L40_P` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L20_A` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L40_A` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L20_S` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L40_S` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L10_Tier_Points` | official game-log feature | 178/439 | 5 | snapshot check passed |
| `L10_Rate_Points` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L10_Diff_Points` | official game-log feature | 160/439 | 16 | snapshot check passed |
| `L20_Tier_Points` | official game-log feature | 178/439 | 3 | snapshot check passed |
| `L20_Rate_Points` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L20_Diff_Points` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L40_Tier_Points` | official game-log feature | 178/439 | 1 | snapshot check passed |
| `L40_Rate_Points` | official game-log feature | 178/439 | 15 | snapshot check passed |
| `L40_Diff_Points` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L10_Tier_Assists` | official game-log feature | 178/439 | 5 | snapshot check passed |
| `L10_Rate_Assists` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L10_Diff_Assists` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L20_Tier_Assists` | official game-log feature | 178/439 | 3 | snapshot check passed |
| `L20_Rate_Assists` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L20_Diff_Assists` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L40_Tier_Assists` | official game-log feature | 178/439 | 1 | snapshot check passed |
| `L40_Rate_Assists` | official game-log feature | 178/439 | 11 | snapshot check passed |
| `L40_Diff_Assists` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L10_Tier_SOG` | official game-log feature | 178/439 | 3 | snapshot check passed |
| `L10_Rate_SOG` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L10_Diff_SOG` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L20_Tier_SOG` | official game-log feature | 178/439 | 1 | snapshot check passed |
| `L20_Rate_SOG` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L20_Diff_SOG` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `L40_Tier_SOG` | official game-log feature | 178/439 | 1 | snapshot check passed |
| `L40_Rate_SOG` | official game-log feature | 178/439 | 26 | snapshot check passed |
| `L40_Diff_SOG` | official game-log feature | 0/439 | 0 | empty; empty in this slate |
| `i5v5_points60` | season stat/derived context | 436/439 | 314 | snapshot check passed |
| `i5v5_primaryAssists60` | season stat/derived context | 436/439 | 170 | snapshot check passed |
| `i5v5_shotAssists60` | season stat/derived context | 0/439 | 0 | empty; empty in this slate |
| `i5v5_iCF60` | season stat/derived context | 0/439 | 0 | empty; empty in this slate |
| `Assist_Volume` | decision heuristic | 436/439 | 11 | snapshot check passed |
| `opp_5v5_xGA60` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `opp_5v5_HDCA60` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `opp_5v5_SlotSA60` | season stat/derived context | 436/439 | 1 | snapshot check passed |
| `Opp_DefWeak` | season stat/derived context | 436/439 | 21 | snapshot check passed |
| `v2_player_stability` | season stat/derived context | 436/439 | 314 | snapshot check passed |
| `v2_defense_vulnerability` | season stat/derived context | 436/439 | 22 | snapshot check passed |
| `Exp_P_10` | decision heuristic | 436/439 | 344 | snapshot check passed |
| `Exp_G_10` | decision heuristic | 436/439 | 257 | snapshot check passed |
| `Exp_S_10` | decision heuristic | 178/439 | 80 | snapshot check passed |
| `Reg_Gap_P10` | decision heuristic | 178/439 | 160 | snapshot check passed |
| `Reg_Gap_G10` | decision heuristic | 178/439 | 145 | snapshot check passed |
| `Reg_Gap_S10` | decision heuristic | 178/439 | 129 | snapshot check passed |
| `Reg_Heat_P` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Reg_Heat_G` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Reg_Heat_S` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Exp_A_10` | decision heuristic | 436/439 | 219 | snapshot check passed |
| `Reg_Gap_A10` | decision heuristic | 178/439 | 133 | snapshot check passed |
| `Reg_Heat_A` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Matrix_Points` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Matrix_SOG` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Matrix_Goal` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Matrix_Assists` | decision heuristic | 436/439 | 3 | snapshot check passed |
| `Conf_Points` | decision heuristic | 436/439 | 75 | snapshot check passed |
| `Conf_SOG` | decision heuristic | 436/439 | 76 | snapshot check passed |
| `Conf_Goal` | decision heuristic | 436/439 | 85 | snapshot check passed |
| `Conf_Assists` | decision heuristic | 436/439 | 81 | snapshot check passed |
| `Best_Market` | decision heuristic | 436/439 | 4 | snapshot check passed |
| `Best_Conf` | decision heuristic | 436/439 | 78 | snapshot check passed |
| `Drought_P` | official game-log feature | 178/439 | 7 | snapshot check passed |
| `Drought_A` | official game-log feature | 178/439 | 9 | snapshot check passed |
| `Drought_G` | official game-log feature | 178/439 | 9 | snapshot check passed |
| `Drought_SOG` | official game-log feature | 178/439 | 9 | snapshot check passed |
| `PPP10_total` | official game-log feature | 178/439 | 7 | snapshot check passed |
| `Drought_PPP` | official game-log feature | 178/439 | 9 | snapshot check passed |
| `PP_Tier` | power-play data/heuristic | 436/439 | 3 | snapshot check passed |
| `PP_Path` | power-play data/heuristic | 436/439 | 4 | snapshot check passed |
| `PP_TeamShare_pct` | power-play data/heuristic | 436/439 | 161 | snapshot check passed |
| `PP_TOI_stability` | power-play data/heuristic | 0/439 | 0 | empty; empty in this slate |
| `PP_Env_Score` | power-play data/heuristic | 436/439 | 22 | snapshot check passed |
| `PP_BOOST` | power-play data/heuristic | 18/439 | 1 | snapshot check passed |
| `PP_Role` | power-play data/heuristic | 436/439 | 3 | snapshot check passed |
| `PP_TOI` | power-play data/heuristic | 436/439 | 330 | snapshot check passed |
| `PP_TOI_min` | power-play data/heuristic | 436/439 | 284 | snapshot check passed |
| `PP_TOI_per_game` | power-play data/heuristic | 436/439 | 330 | snapshot check passed |
| `PP_TOI_Share` | power-play data/heuristic | 436/439 | 383 | snapshot check passed |
| `PP_TOI_Pct_Game` | power-play data/heuristic | 436/439 | 161 | snapshot check passed |
| `PP_TOI_Pct` | power-play data/heuristic | 436/439 | 161 | snapshot check passed |
| `PP_Points60` | power-play data/heuristic | 436/439 | 215 | snapshot check passed |
| `PP_iXG60` | power-play data/heuristic | 382/439 | 212 | snapshot check passed |
| `PP_iXA60` | power-play data/heuristic | 382/439 | 108 | snapshot check passed |
| `Team_PP_xGF60` | power-play data/heuristic | 436/439 | 22 | snapshot check passed |
| `Opp_PK_xGA60` | power-play data/heuristic | 436/439 | 22 | snapshot check passed |
| `PP_Matchup` | power-play data/heuristic | 436/439 | 22 | snapshot check passed |
| `Assist_Dagger` | decision heuristic | 436/439 | 288 | snapshot check passed |
| `Assist_PP_Proof` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Line` | empty grading placeholder | 0/439 | 0 | empty; empty in this slate |
| `Odds` | empty grading placeholder | 0/439 | 0 | empty; empty in this slate |
| `Result` | empty grading placeholder | 0/439 | 0 | empty; empty in this slate |
| `Best_Drought` | decision heuristic | 178/439 | 23 | snapshot check passed |
| `Play_Tag` | season stat/derived context | 19/439 | 1 | snapshot check passed |
| `Plays_Points` | decision heuristic | 436/439 | 1 | snapshot check passed |
| `Plays_Goal` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Goal_ProofCount` | decision heuristic | 436/439 | 4 | snapshot check passed |
| `Goal_Why` | decision heuristic | 19/439 | 3 | snapshot check passed |
| `Plays_Assists` | decision heuristic | 436/439 | 1 | snapshot check passed |
| `Assist_ProofCount` | decision heuristic | 436/439 | 1 | snapshot check passed |
| `Assist_Why` | decision heuristic | 0/439 | 0 | empty; empty in this slate |
| `Plays_SOG` | decision heuristic | 436/439 | 1 | snapshot check passed |
| `SOG_ProofCount` | decision heuristic | 436/439 | 1 | snapshot check passed |
| `SOG_Why` | decision heuristic | 0/439 | 0 | empty; empty in this slate |
| `BDL_SOG_Line_1` | captured book quote | 251/439 | 1 | snapshot check passed |
| `BDL_SOG_Odds_1` | captured book quote | 251/439 | 111 | snapshot check passed |
| `BDL_SOG_Book_1` | captured book quote | 251/439 | 10 | snapshot check passed |
| `BDL_SOG_Line_2` | captured book quote | 191/439 | 1 | snapshot check passed |
| `BDL_SOG_Odds_2` | captured book quote | 191/439 | 90 | snapshot check passed |
| `BDL_SOG_Book_2` | captured book quote | 191/439 | 10 | snapshot check passed |
| `BDL_SOG_Line_3` | captured book quote | 190/439 | 1 | snapshot check passed |
| `BDL_SOG_Odds_3` | captured book quote | 190/439 | 68 | snapshot check passed |
| `BDL_SOG_Book_3` | captured book quote | 190/439 | 8 | snapshot check passed |
| `BDL_SOG_Line_4` | captured book quote | 190/439 | 1 | snapshot check passed |
| `BDL_SOG_Odds_4` | captured book quote | 190/439 | 52 | snapshot check passed |
| `BDL_SOG_Book_4` | captured book quote | 190/439 | 5 | snapshot check passed |
| `BDL_SOG_Line` | captured book quote | 251/439 | 3 | snapshot check passed |
| `BDL_SOG_Odds` | captured book quote | 251/439 | 85 | snapshot check passed |
| `BDL_SOG_Book` | captured book quote | 251/439 | 10 | snapshot check passed |
| `BDL_Points_Line_1` | captured book quote | 286/439 | 1 | snapshot check passed |
| `BDL_Points_Odds_1` | captured book quote | 286/439 | 105 | snapshot check passed |
| `BDL_Points_Book_1` | captured book quote | 286/439 | 11 | snapshot check passed |
| `BDL_Points_Line_2` | captured book quote | 251/439 | 1 | snapshot check passed |
| `BDL_Points_Odds_2` | captured book quote | 251/439 | 67 | snapshot check passed |
| `BDL_Points_Book_2` | captured book quote | 251/439 | 10 | snapshot check passed |
| `BDL_Points_Line_3` | captured book quote | 230/439 | 1 | snapshot check passed |
| `BDL_Points_Odds_3` | captured book quote | 230/439 | 35 | snapshot check passed |
| `BDL_Points_Book_3` | captured book quote | 230/439 | 7 | snapshot check passed |
| `BDL_Points_Line_4` | captured book quote | 229/439 | 1 | snapshot check passed |
| `BDL_Points_Odds_4` | captured book quote | 229/439 | 33 | snapshot check passed |
| `BDL_Points_Book_4` | captured book quote | 229/439 | 2 | snapshot check passed |
| `BDL_Points_Line` | captured book quote | 286/439 | 2 | snapshot check passed |
| `BDL_Points_Odds` | captured book quote | 286/439 | 96 | snapshot check passed |
| `BDL_Points_Book` | captured book quote | 286/439 | 11 | snapshot check passed |
| `BDL_Assists_Line_1` | captured book quote | 259/439 | 1 | snapshot check passed |
| `BDL_Assists_Odds_1` | captured book quote | 259/439 | 77 | snapshot check passed |
| `BDL_Assists_Book_1` | captured book quote | 259/439 | 12 | snapshot check passed |
| `BDL_Assists_Line_2` | captured book quote | 251/439 | 1 | snapshot check passed |
| `BDL_Assists_Odds_2` | captured book quote | 251/439 | 48 | snapshot check passed |
| `BDL_Assists_Book_2` | captured book quote | 251/439 | 7 | snapshot check passed |
| `BDL_Assists_Line_3` | captured book quote | 53/439 | 1 | snapshot check passed |
| `BDL_Assists_Odds_3` | captured book quote | 53/439 | 18 | snapshot check passed |
| `BDL_Assists_Book_3` | captured book quote | 53/439 | 3 | snapshot check passed |
| `BDL_Assists_Line_4` | captured book quote | 6/439 | 1 | snapshot check passed |
| `BDL_Assists_Odds_4` | captured book quote | 6/439 | 5 | snapshot check passed |
| `BDL_Assists_Book_4` | captured book quote | 6/439 | 1 | snapshot check passed |
| `BDL_Assists_Line` | captured book quote | 259/439 | 2 | snapshot check passed |
| `BDL_Assists_Odds` | captured book quote | 259/439 | 76 | snapshot check passed |
| `BDL_Assists_Book` | captured book quote | 259/439 | 12 | snapshot check passed |
| `BDL_Goal_Line_1` | captured book quote | 392/439 | 1 | snapshot check passed |
| `BDL_Goal_Odds_1` | captured book quote | 392/439 | 83 | snapshot check passed |
| `BDL_Goal_Book_1` | captured book quote | 392/439 | 11 | snapshot check passed |
| `BDL_Goal_Line_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Line_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Line_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Odds_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Book_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_Goal_Line` | captured book quote | 392/439 | 1 | snapshot check passed |
| `BDL_Goal_Odds` | captured book quote | 392/439 | 83 | snapshot check passed |
| `BDL_Goal_Book` | captured book quote | 392/439 | 11 | snapshot check passed |
| `BDL_ATG_Line_1` | captured book quote | 396/439 | 1 | snapshot check passed |
| `BDL_ATG_Odds_1` | captured book quote | 396/439 | 87 | snapshot check passed |
| `BDL_ATG_Book_1` | captured book quote | 396/439 | 12 | snapshot check passed |
| `BDL_ATG_Line_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_2` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Line_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Line_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Odds_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Book_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_ATG_Line` | captured book quote | 396/439 | 1 | snapshot check passed |
| `BDL_ATG_Odds` | captured book quote | 396/439 | 87 | snapshot check passed |
| `BDL_ATG_Book` | captured book quote | 396/439 | 12 | snapshot check passed |
| `Goal_Odds_Source` | captured book quote | 392/439 | 1 | snapshot check passed |
| `Goal_Available_Books` | captured book quote | 392/439 | 47 | snapshot check passed |
| `ATG_Odds_Source` | captured book quote | 396/439 | 2 | snapshot check passed |
| `ATG_Available_Books` | captured book quote | 396/439 | 55 | snapshot check passed |
| `SOG_Odds_Source` | captured book quote | 251/439 | 1 | snapshot check passed |
| `SOG_Available_Books` | captured book quote | 251/439 | 56 | snapshot check passed |
| `BDL_PPP_Line_1` | captured book quote | 200/439 | 1 | snapshot check passed |
| `BDL_PPP_Odds_1` | captured book quote | 200/439 | 60 | snapshot check passed |
| `BDL_PPP_Book_1` | captured book quote | 200/439 | 5 | snapshot check passed |
| `BDL_PPP_Line_2` | captured book quote | 104/439 | 1 | snapshot check passed |
| `BDL_PPP_Odds_2` | captured book quote | 104/439 | 19 | snapshot check passed |
| `BDL_PPP_Book_2` | captured book quote | 104/439 | 2 | snapshot check passed |
| `BDL_PPP_Line_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Odds_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Book_3` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Line_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Odds_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Book_4` | captured book quote | 0/439 | 0 | empty; empty in this slate |
| `BDL_PPP_Line` | captured book quote | 201/439 | 1 | snapshot check passed |
| `BDL_PPP_Odds` | captured book quote | 201/439 | 60 | snapshot check passed |
| `BDL_PPP_Book` | captured book quote | 201/439 | 6 | snapshot check passed |
| `PPP_Odds_Source` | captured book quote | 200/439 | 1 | snapshot check passed |
| `PPP_Available_Books` | captured book quote | 200/439 | 22 | snapshot check passed |
| `Points_Odds_Source` | captured book quote | 286/439 | 2 | snapshot check passed |
| `Points_Available_Books` | captured book quote | 286/439 | 69 | snapshot check passed |
| `Assists_Odds_Source` | captured book quote | 259/439 | 2 | snapshot check passed |
| `Assists_Available_Books` | captured book quote | 259/439 | 70 | snapshot check passed |
| `Odds_Name_Ambiguous` | captured book quote | 436/439 | 2 | snapshot check passed |
| `Odds_Checked_UTC` | captured book quote | 436/439 | 1 | snapshot check passed |
| `BDL_EV_ENGINE` | captured book quote | 436/439 | 1 | snapshot check passed |
| `SOG_Line` | selected book quote | 254/439 | 3 | snapshot check passed |
| `SOG_Odds_Over` | selected book quote | 254/439 | 86 | snapshot check passed |
| `SOG_Book` | selected book quote | 254/439 | 13 | snapshot check passed |
| `SOG_ModelSrc` | model/price calculation | 436/439 | 2 | snapshot check passed |
| `SOG_mu` | model/price calculation | 436/439 | 81 | snapshot check passed |
| `SOG_p_model_over` | model/price calculation | 251/439 | 97 | snapshot check passed |
| `SOG_Model%` | model/price calculation | 251/439 | 92 | snapshot check passed |
| `SOG_p_imp_over` | model/price calculation | 251/439 | 85 | snapshot check passed |
| `SOG_EVpct_over` | model/price calculation | 251/439 | 203 | snapshot check passed |
| `SOG_EV%` | model/price calculation | 251/439 | 189 | snapshot check passed |
| `SOG_p_model_over_1` | model/price calculation | 251/439 | 77 | snapshot check passed |
| `SOG_Model%_1` | model/price calculation | 251/439 | 77 | snapshot check passed |
| `SOG_EVpct_over_1` | model/price calculation | 251/439 | 202 | snapshot check passed |
| `SOG_EV%_1` | model/price calculation | 251/439 | 189 | snapshot check passed |
| `SOG_p_model_over_2` | model/price calculation | 191/439 | 72 | snapshot check passed |
| `SOG_Model%_2` | model/price calculation | 191/439 | 72 | snapshot check passed |
| `SOG_EVpct_over_2` | model/price calculation | 191/439 | 169 | snapshot check passed |
| `SOG_EV%_2` | model/price calculation | 191/439 | 164 | snapshot check passed |
| `SOG_p_model_over_3` | model/price calculation | 190/439 | 72 | snapshot check passed |
| `SOG_Model%_3` | model/price calculation | 190/439 | 71 | snapshot check passed |
| `SOG_EVpct_over_3` | model/price calculation | 190/439 | 158 | snapshot check passed |
| `SOG_EV%_3` | model/price calculation | 190/439 | 153 | snapshot check passed |
| `SOG_p_model_over_4` | model/price calculation | 190/439 | 72 | snapshot check passed |
| `SOG_Model%_4` | model/price calculation | 190/439 | 71 | snapshot check passed |
| `SOG_EVpct_over_4` | model/price calculation | 190/439 | 149 | snapshot check passed |
| `SOG_EV%_4` | model/price calculation | 190/439 | 144 | snapshot check passed |
| `Points_Line` | selected book quote | 290/439 | 2 | snapshot check passed |
| `Points_Odds_Over` | selected book quote | 290/439 | 96 | snapshot check passed |
| `Points_Book` | selected book quote | 290/439 | 13 | snapshot check passed |
| `Points_ModelSrc` | model/price calculation | 436/439 | 2 | snapshot check passed |
| `Points_mu` | model/price calculation | 436/439 | 343 | snapshot check passed |
| `Points_p_model_over` | model/price calculation | 286/439 | 253 | snapshot check passed |
| `Points_Model%` | model/price calculation | 286/439 | 221 | snapshot check passed |
| `Points_p_imp_over` | model/price calculation | 286/439 | 96 | snapshot check passed |
| `Points_EVpct_over` | model/price calculation | 286/439 | 285 | snapshot check passed |
| `Points_EV%` | model/price calculation | 286/439 | 259 | snapshot check passed |
| `Points_p_model_over_1` | model/price calculation | 286/439 | 252 | snapshot check passed |
| `Points_Model%_1` | model/price calculation | 286/439 | 223 | snapshot check passed |
| `Points_EVpct_over_1` | model/price calculation | 286/439 | 285 | snapshot check passed |
| `Points_EV%_1` | model/price calculation | 286/439 | 261 | snapshot check passed |
| `Points_p_model_over_2` | model/price calculation | 251/439 | 228 | snapshot check passed |
| `Points_Model%_2` | model/price calculation | 251/439 | 173 | snapshot check passed |
| `Points_EVpct_over_2` | model/price calculation | 251/439 | 251 | snapshot check passed |
| `Points_EV%_2` | model/price calculation | 251/439 | 244 | snapshot check passed |
| `Points_p_model_over_3` | model/price calculation | 230/439 | 213 | snapshot check passed |
| `Points_Model%_3` | model/price calculation | 230/439 | 116 | snapshot check passed |
| `Points_EVpct_over_3` | model/price calculation | 230/439 | 229 | snapshot check passed |
| `Points_EV%_3` | model/price calculation | 230/439 | 214 | snapshot check passed |
| `Points_p_model_over_4` | model/price calculation | 229/439 | 211 | snapshot check passed |
| `Points_Model%_4` | model/price calculation | 229/439 | 77 | snapshot check passed |
| `Points_EVpct_over_4` | model/price calculation | 229/439 | 227 | snapshot check passed |
| `Points_EV%_4` | model/price calculation | 229/439 | 191 | snapshot check passed |
| `Assists_Line` | selected book quote | 263/439 | 2 | snapshot check passed |
| `Assists_Odds_Over` | selected book quote | 263/439 | 76 | snapshot check passed |
| `Assists_Book` | selected book quote | 263/439 | 12 | snapshot check passed |
| `Assists_ModelSrc` | model/price calculation | 436/439 | 2 | snapshot check passed |
| `Assists_mu` | model/price calculation | 436/439 | 218 | snapshot check passed |
| `Assists_p_model_over` | model/price calculation | 259/439 | 182 | snapshot check passed |
| `Assists_Model%` | model/price calculation | 259/439 | 163 | snapshot check passed |
| `Assists_p_imp_over` | model/price calculation | 259/439 | 76 | snapshot check passed |
| `Assists_EVpct_over` | model/price calculation | 259/439 | 246 | snapshot check passed |
| `Assists_EV%` | model/price calculation | 259/439 | 226 | snapshot check passed |
| `Assists_p_model_over_1` | model/price calculation | 259/439 | 182 | snapshot check passed |
| `Assists_Model%_1` | model/price calculation | 259/439 | 163 | snapshot check passed |
| `Assists_EVpct_over_1` | model/price calculation | 259/439 | 246 | snapshot check passed |
| `Assists_EV%_1` | model/price calculation | 259/439 | 226 | snapshot check passed |
| `Assists_p_model_over_2` | model/price calculation | 251/439 | 180 | snapshot check passed |
| `Assists_Model%_2` | model/price calculation | 251/439 | 123 | snapshot check passed |
| `Assists_EVpct_over_2` | model/price calculation | 251/439 | 234 | snapshot check passed |
| `Assists_EV%_2` | model/price calculation | 251/439 | 222 | snapshot check passed |
| `Assists_p_model_over_3` | model/price calculation | 53/439 | 51 | snapshot check passed |
| `Assists_Model%_3` | model/price calculation | 53/439 | 42 | snapshot check passed |
| `Assists_EVpct_over_3` | model/price calculation | 53/439 | 52 | snapshot check passed |
| `Assists_EV%_3` | model/price calculation | 53/439 | 51 | snapshot check passed |
| `Assists_p_model_over_4` | model/price calculation | 6/439 | 6 | snapshot check passed |
| `Assists_Model%_4` | model/price calculation | 6/439 | 6 | snapshot check passed |
| `Assists_EVpct_over_4` | model/price calculation | 6/439 | 6 | snapshot check passed |
| `Assists_EV%_4` | model/price calculation | 6/439 | 6 | snapshot check passed |
| `Goal_Line` | selected book quote | 401/439 | 1 | snapshot check passed |
| `Goal_Odds_Over` | selected book quote | 401/439 | 84 | snapshot check passed |
| `Goal_Book` | selected book quote | 401/439 | 12 | snapshot check passed |
| `Goal_ModelSrc` | model/price calculation | 436/439 | 2 | snapshot check passed |
| `Goal_mu` | model/price calculation | 436/439 | 290 | snapshot check passed |
| `Goal_p_model_over` | model/price calculation | 392/439 | 277 | snapshot check passed |
| `Goal_Model%` | model/price calculation | 392/439 | 140 | snapshot check passed |
| `Goal_p_imp_over` | model/price calculation | 392/439 | 83 | snapshot check passed |
| `Goal_EVpct_over` | model/price calculation | 392/439 | 389 | snapshot check passed |
| `Goal_EV%` | model/price calculation | 392/439 | 362 | snapshot check passed |
| `Goal_p_model_over_1` | model/price calculation | 392/439 | 277 | snapshot check passed |
| `Goal_Model%_1` | model/price calculation | 392/439 | 140 | snapshot check passed |
| `Goal_EVpct_over_1` | model/price calculation | 392/439 | 389 | snapshot check passed |
| `Goal_EV%_1` | model/price calculation | 392/439 | 362 | snapshot check passed |
| `Goal_p_model_over_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_Model%_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EV%_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_p_model_over_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_Model%_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EV%_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_p_model_over_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_Model%_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EVpct_over_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Goal_EV%_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_Line` | selected book quote | 396/439 | 1 | snapshot check passed |
| `ATG_Odds_Over` | selected book quote | 396/439 | 87 | snapshot check passed |
| `ATG_Book` | selected book quote | 396/439 | 12 | snapshot check passed |
| `ATG_ModelSrc` | model/price calculation | 436/439 | 2 | snapshot check passed |
| `ATG_mu` | model/price calculation | 436/439 | 298 | snapshot check passed |
| `ATG_p_model_over` | model/price calculation | 396/439 | 287 | snapshot check passed |
| `ATG_Model%` | model/price calculation | 396/439 | 176 | snapshot check passed |
| `ATG_p_imp_over` | model/price calculation | 396/439 | 87 | snapshot check passed |
| `ATG_EVpct_over` | model/price calculation | 396/439 | 394 | snapshot check passed |
| `ATG_EV%` | model/price calculation | 396/439 | 350 | snapshot check passed |
| `ATG_p_model_over_1` | model/price calculation | 396/439 | 287 | snapshot check passed |
| `ATG_Model%_1` | model/price calculation | 396/439 | 176 | snapshot check passed |
| `ATG_EVpct_over_1` | model/price calculation | 396/439 | 394 | snapshot check passed |
| `ATG_EV%_1` | model/price calculation | 396/439 | 350 | snapshot check passed |
| `ATG_p_model_over_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_Model%_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EV%_2` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_p_model_over_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_Model%_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EV%_3` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_p_model_over_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_Model%_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EVpct_over_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `ATG_EV%_4` | model/price calculation | 0/439 | 0 | empty; empty in this slate |
| `Plays_EV_SOG` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Plays_EV_Points` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Plays_EV_Goal` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Plays_EV_ATG` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Plays_EV_Assists` | decision heuristic | 436/439 | 2 | snapshot check passed |
| `Player_SOG_Share_Proxy` | season stat/derived context | 178/439 | 62 | snapshot check passed |
| `Outcome` | empty grading placeholder | 0/439 | 0 | empty; empty in this slate |
| `Roster_Status` | availability/roster | 439/439 | 1 | snapshot check passed |
| `Roster_Check_UTC` | availability/roster | 439/439 | 1 | snapshot check passed |
| `Roster_Watch` | availability/roster | 439/439 | 2 | snapshot check passed |
| `Fired_Moves` | decision heuristic | 439/439 | 33 | snapshot check passed |
| `Model_Stats_Season` | season stat/derived context | 439/439 | 2 | snapshot check passed |
| `Available` | availability/roster | 3/439 | 1 | snapshot check passed |
| `Move_Kit_Version` | season stat/derived context | 439/439 | 1 | snapshot check passed |
