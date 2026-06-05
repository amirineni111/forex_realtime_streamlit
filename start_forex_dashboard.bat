@echo off
cd /d "%~dp0"
echo Starting Forex Trading Dashboard...
echo Open http://localhost:8501 in your browser
call venv\Scripts\activate.bat
venv\Scripts\streamlit run app.py --server.port 8501
pause
