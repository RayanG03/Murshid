@echo off
rem Starts the whole project: Qdrant (Docker) + server, then opens the UI when ready.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo [1/3] Starting Qdrant...
docker start qdrant >nul 2>&1
if errorlevel 1 echo     Qdrant did not start - is Docker Desktop open? Search will use local files.

echo [2/3] The browser opens by itself when the server is ready (about a minute)...
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 150;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2 | Out-Null; Start-Process http://127.0.0.1:8000; break}catch{Start-Sleep 2}}"

echo [3/3] Starting the server - close this window or press Ctrl+C to stop it.
.venv\Scripts\python.exe app.py
pause
