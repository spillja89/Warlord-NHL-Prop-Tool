GitHub Actions auto-slate setup

1) Copy this workflow file into your repo:
   .github/workflows/daily_slate.yml

2) In your GitHub repo, add secrets (if your model needs them):
   Settings -> Secrets and variables -> Actions -> New repository secret
   - BALLDONTLIE_API_KEY
   - SPORTSDATAIO_KEY

3) The workflow runs every day at ~06:05 America/Chicago (12:05 UTC).
   You can also run it manually:
   Actions -> "Build daily slate (tracker_latest)" -> Run workflow

4) Streamlit Cloud will pick up the new commit and your app will auto-load:
   output/tracker_latest.csv
