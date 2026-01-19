Warlord NHL Prop Tool — EV (SOG) Fix Pack

What’s fixed
- nhl_edge.py: clean, syntax-checked, and the SOG odds/EV merge is inserted in the correct place
  (right before the tracker CSV is written), using the tracker dataframe.

What you need
- Set env var: BALLDONTLIE_API_KEY

PowerShell (persistent):
  setx BALLDONTLIE_API_KEY "YOUR_KEY"
  # close/reopen terminal

Run
  python -u nhl_edge.py --date 2026-01-17
  streamlit run app.py

Verify EV columns exist in latest tracker CSV:
  python -c "import glob,os,pandas as pd; p=max(glob.glob(r'output\tracker_*.csv'), key=os.path.getmtime); df=pd.read_csv(p); print([c for c in df.columns if 'SOG_' in c or 'EV' in c or 'Odds' in c or 'Book' in c])"
