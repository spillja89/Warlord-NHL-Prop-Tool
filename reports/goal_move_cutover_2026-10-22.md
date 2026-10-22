# Goal move cutover map — 2026–27 regular season

This is a **name and rule replacement**, not a pooling of 2025–26 and 2026–27 results. Every record below is a graded 2026–27 Goal over 0.5 player-game from the saved pregame tracker. All new rules require a posted Goal 0.5 price, Matrix_Goal Green, and Conf_Goal at least 85. A move percentage is an in-sample historical result, not tonight's probability or a validated edge against the book.

| Live card name | Research branch / additional gates | 2026–27 goals | Replaces old name |
|---|---|---:|---|
| Base Attack | TOI/game ≥19; SOG conf ≥80 | 60/116 | Base Attack |
| Shooter's Command | Base; TOI percentile ≥88; 5v5 SOG share ≥15%; Goal μ ≥.45 | 17/21 | **NEW** |
| Power Finisher | Base; TOI percentile ≥88; PP iXG/60 ≥1.6; Goal μ ≥.45 | 23/31 | **NEW** |
| Press the Attack | TOI percentile ≥84; StarScore ≥88 | 46/82 | Press the Attack |
| Fury Shredder | TOI percentile ≥84; PP iXG/60 ≥1.2 | 44/77 | Fury Shredder |
| Finisher Strike | Talent multiplier ≥1.2; last-five goals/game ≥.4; TOI/game ≥19 | 41/73 | Finisher Strike |
| Enraged Finisher | Press the Attack; last-five goals/game ≥.6 | 33/51 | Enraged Finisher |
| Bloodlust | Fury Shredder; Goal μ ≥.5 | 22/32 | Bloodlust |
| Enraged Fury Shredder | Finisher Strike; last-five xG/game ≥.55 | 19/29 | Enraged Fury Shredder |
| Bloodlusted Surge | Enraged Finisher; team goals/game, last five ≥3.2 | 26/37 | Bloodlusted Surge |
| Open Lanes Assault | Points μ ≥1.25; last-five SOG/game ≥3.4 | 36/60 | Open Lanes Assault |
| Warlord Smash Attack | Points μ ≥1.25; last-five xG/game ≥.55 | 25/38 | Warlord Smash Attack |
| Berserker Aggression | TOI/game ≥21; Goal μ ≥.45 | 26/38 | Berserker Aggression |
| Warlord Smash Attack (Crit) | Open Lanes Assault; Goal μ ≥.45 | 30/45 | Warlord Smash Attack (Crit) |
| Expected Shot Barrage | Warlord Smash Attack; last-ten shots ≥19 | 21/29 | **NEW** |
| Tyr’s Wrath Unleashed | Berserker Aggression; PP iXG/60 ≥1.6 | 24/32 | Tyr’s Wrath Unleashed |
| FOR VALHALLA! | Warlord Smash Attack (Crit); opponent PK xGA/60 ≥6 | 26/34 | FOR VALHALLA! |

The research branch names are stored separately from the familiar card names in `goal_moves_current.py`, so the exact graded records remain traceable. The three new names above have no old equivalent. All 17 records were replayed against the saved 2026–27 graded tracker and matched exactly.

**Old names retired:** Weakened Armor, Open Lanes, Fallen Guardian, Armor Sunder, Armor Annihilation, Press the Attack (Crit), Bloodlusted. Their opponent-only or goalie-driven rules no longer fire Goal cards. The old points-confidence S/A/B split and its cumulative 81+/82+/84+ records are removed from Goal card selection and display.

The new named rules use the saved pregame tracker values. Recent goals and xG come from MoneyPuck's 2026 regular-season game-by-game skater download, using only games before the slate date. The NHL regular-season log provides a recent-goals fallback. If recent xG is unavailable, the xG-dependent moves do not fire. Published historical move tags remain frozen on the original slates for grading; new runs carry a distinct kit version.
