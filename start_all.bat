@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist logs mkdir logs

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PYTHONW_EXE=%~dp0.venv\Scripts\pythonw.exe"
set "YOLO_SCRIPT=%~dp0yolo_server.py"
set "LAUNCHER_SCRIPT=%~dp0launcher.py"
set "YOLO_LOG=%~dp0logs\yolo.log"
set "LAUNCHER_LOG=%~dp0logs\launcher.log"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist "%PYTHON_EXE%" (
    echo [START_ALL] Missing Python: "%PYTHON_EXE%"
    goto END
)

if not exist "%PYTHONW_EXE%" (
    echo [START_ALL] Missing Pythonw: "%PYTHONW_EXE%"
    goto END
)

if not exist "%YOLO_SCRIPT%" (
    echo [START_ALL] Missing YOLO script: "%YOLO_SCRIPT%"
    goto END
)

if not exist "%LAUNCHER_SCRIPT%" (
    echo [START_ALL] Missing launcher script: "%LAUNCHER_SCRIPT%"
    goto END
)

echo [START_ALL] Starting YOLO server...
start "YOLO_SERVER" /min cmd /c ""%PYTHON_EXE%" "%YOLO_SCRIPT%" >> "%YOLO_LOG%" 2>&1"

set /a COUNT=0
set /a MAX_WAIT=20

:WAIT_YOLO
curl.exe -s http://127.0.0.1:8000/health >nul 2>&1
if %errorlevel%==0 goto START_LAUNCHER

timeout /t 2 /nobreak >nul
set /a COUNT+=1

if %COUNT% GEQ %MAX_WAIT% (
    echo [START_ALL] YOLO health check timeout. See "%YOLO_LOG%"
    goto END
)

goto WAIT_YOLO

:START_LAUNCHER
echo [START_ALL] Starting launcher...
start "SAFE_DRIVE_LAUNCHER" /min cmd /c ""%PYTHONW_EXE%" "%LAUNCHER_SCRIPT%" >> "%LAUNCHER_LOG%" 2>&1"

echo [START_ALL] Done.
goto END

:END
endlocal
exit /b