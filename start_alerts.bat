@echo off
cd /d "%~dp0"
echo Starting real-time forex alerts (scans after every signal-bar close, Sun 5pm - Fri 5pm ET)
echo New alerts are pushed to the ntfy topic in .env. Close this window to stop.
venv\Scripts\python scripts\run_alerts.py
pause
