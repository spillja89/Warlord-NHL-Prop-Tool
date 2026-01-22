# ==========================
# NHL EDGE - One Click Runner
# ==========================

Write-Host "Starting NHL Edge..." -ForegroundColor Cyan

# --- CONFIG ---
$PROJECT_DIR = "G:\My Drive\Nhl Prop Tool"
$VENV_ACTIVATE = "C:\Users\spill\.venvs\nhl_prop_tool\Scripts\Activate.ps1"

# --- GO TO PROJECT ---
Set-Location $PROJECT_DIR

# --- ACTIVATE VENV ---
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& $VENV_ACTIVATE

# --- RUN DATA PIPELINE ---
Write-Host "Running nhl_edge.py..." -ForegroundColor Green
python nhl_edge.py

# --- START UI ---
Write-Host "Launching Streamlit app..." -ForegroundColor Green
streamlit run app.py
