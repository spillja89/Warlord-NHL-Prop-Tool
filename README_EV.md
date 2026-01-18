EV Patch (SOG) — Warlord NHL Prop Tool

What this includes
- nhl_edge.py: merges SOG odds + EV onto the tracker dataframe right before CSV write.
- odds_ev_sog.py: pulls BallDontLie NHL props (shots_on_goal) and computes model p + implied p + EV%.
- app.py: shows the new SOG columns on the SOG page (safe if columns missing).

Required env var
  BALLDONTLIE_API_KEY=<your_key>

Quick run
  python -u nhl_edge.py --date 2026-01-17
  streamlit run app.py
