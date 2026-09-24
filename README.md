# The Warlord's NHL Prop Tool

## The 2026 path

`nhl_edge.py` builds a daily tracker. `odds_ev_bdl.py` adds available book lines. `app.py` loads `output/tracker_latest.csv` and displays the Board, Goals, Assists, Points, SOG, Power Play, and scouting pages. All four named 2026 move kits and their historical samples live in `warlord_moves_2026.py`. The card, sorting, and pregame snapshot use the same rules and names.

The graded rates on move cards describe historical model picks. A pick can fire several moves. **TRACK** marks a small or exploratory subset; it is not a model probability.

## Run locally

Use Python 3.11 or 3.12, install `requirements.txt`, then run `python -m streamlit run app.py`. The app opens the saved tracker without a key. To refresh a slate from inside the app, set `BALLDONTLIE_API_KEY` and `WARLORD_ADMIN_PASSWORD` in environment variables or Streamlit secrets. The owner password unlocks **Run / Refresh slate**.

Local bet logging requires `WARLORD_LEDGER_MODE=local` and the owner password. It writes `output/ledger/betslip.csv`. For private durable cloud logging, configure a PostgreSQL database and set `WARLORD_LEDGER_MODE=postgres`, `WARLORD_LEDGER_DATABASE_URL`, and `WARLORD_ADMIN_PASSWORD` in server-side secrets. The database connection requires TLS; no bet data or database URL is put in the tracker. Keep local mode off on Streamlit Community Cloud because its local files are not durable. Without a configured ledger backend, logging stays locked. Cloud logging still needs a live connection check after a database is provisioned.

Never commit API keys, owner passwords, or a `secrets.toml` file. Set them in the hosting provider's secrets settings. The app is read-only by default when no owner password is configured.

## Daily pipeline

`.github/workflows/daily_slate.yml` is the sole scheduled slate job. It builds a date-stamped tracker and publishes that date's file as `output/tracker_latest.csv` when it has player rows. The app shows a warning when the saved tracker is from an earlier date. An uploaded daily tracker overrides the saved one for that user's session.
Off days do not publish an empty tracker or fail the scheduled job.

The daily job also saves `output/slates/tracker_YYYY-MM-DD.csv` as a frozen input for grading. `freeze_daily_slate.py` updates only games that have not started, so an afternoon game retains its earlier pregame rows when the evening refresh runs. Each row records its freeze time, move kit version, and the exact named moves that fired. The morning job in `.github/workflows/grade_previous_slate.yml` runs at 7 or 8 AM Chicago time. It grades yesterday's slate and retries the prior two dates so postponed or late games can complete. It commits `output/graded/tracker_YYYY-MM-DD_GRADED.csv`, `output/graded/moves_YYYY-MM-DD.csv`, and a summary JSON. The app's **Results** page shows both market outcomes and forward results for the named moves. Older trackers without frozen move tags never have moves assigned retroactively.

Run the grader yourself with `python grade_daily_tracker.py --date YYYY-MM-DD`. It selects the frozen dated tracker, never `tracker_latest.csv`. For an older file, pass `--tracker path/to/tracker.csv --date YYYY-MM-DD`. Grading fetches final NHL box scores and records actual points, assists, goals, and shots for every matched player. It assigns W/L/P only to markets with an actual line; unresolved games and players are never marked as losses. The original tracker is untouched. New trackers include `Game_ID` and `Player_ID` for exact matching; older files use a unique team and name match. Move results use the frozen pregame tags; one player can appear under several moves, so move totals overlap.

## Launch checks

1. Confirm the Streamlit deployment points to the branch containing these files.
2. Configure the API key and owner password in hosting secrets.
3. Run a fresh slate and confirm its date, player count, all four market views, Board feed-health table, and Power Play coverage. Preseason slates may have no posted prop lines; Green players without lines are a watchlist, not graded picks.
4. If cloud bet logging is wanted, configure the PostgreSQL secrets above and save/read one test bet after deployment. Without a database, keep logging locked.
5. Check a morning Results page for unresolved rows and each move's forward record before using those totals in model testing.

The Power Play page is read-only. PP opportunity counts and true game-level PP usage stability are not yet supplied by the tracker; the page shows the PP metrics that are available.
