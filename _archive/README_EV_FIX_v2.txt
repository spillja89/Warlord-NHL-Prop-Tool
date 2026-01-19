EV Fix v2 (SOG) — columns always present

This version ensures the EV-related SOG columns are created even if the odds API returns no games/props.
So your tracker CSV will always include the SOG_* and SOG_EVpct_over columns (may be blank until odds are found).

1) Overwrite these in your repo:
   - odds_ev_sog.py
   - nhl_edge.py
   - app.py

2) Set your key:
   setx BALLDONTLIE_API_KEY "YOUR_KEY"
   (reopen terminal)

3) Run:
   python -u nhl_edge.py --date 2026-01-17

4) Verify columns:
   python -c "import glob,os,pandas as pd; p=max(glob.glob(r'output\tracker_*.csv'), key=os.path.getmtime); df=pd.read_csv(p); print([c for c in df.columns if 'SOG_' in c or 'EV' in c or 'Odds' in c or 'Book' in c])"
