@echo off
REM FastVideoEdit - launcher for the web editor (double-click me).
REM Creates .venv and installs deps on first run, then starts serve.py.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "SENTINEL=%ROOT%.venv\.fve_installed"

if not exist "%PY%" (
  echo Creating .venv ...
  py -3.12 -m venv "%ROOT%.venv" 2>nul || py -m venv "%ROOT%.venv" 2>nul || python -m venv "%ROOT%.venv"
)

if not exist "%SENTINEL%" (
  "%PY%" -m pip install --upgrade pip
  echo Installing dependencies, please wait ...
  "%PY%" -m pip install -r "%ROOT%requirements.txt"
  if errorlevel 1 (
    echo.
    echo pip install FAILED - fix the error above and run again. & pause & exit /b 1
  )
  echo ok> "%SENTINEL%"
)

where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not found on PATH. Install: winget install Gyan.FFmpeg (open a new terminal after).

"%PY%" "%ROOT%serve.py" %*
echo.
pause
