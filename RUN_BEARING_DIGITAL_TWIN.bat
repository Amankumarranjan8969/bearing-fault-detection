@echo off
setlocal EnableDelayedExpansion
title Bearing Fault Detection - Enterprise Platform
color 0A

echo ===============================================================================
echo        BEARING FAULT DETECTION - ENTERPRISE PLATFORM
echo               Physics-Informed AI - RAG Diagnostics - 3D Twin
echo ===============================================================================
echo.

:: 1. Auto-detect project root directory
set "PROJECT_DIR="

if exist "%~dp0backend\main.py" (
    set "PROJECT_DIR=%~dp0"
)

if not defined PROJECT_DIR (
    if exist "D:\FULL WORK FINAL\bearing-digital-twin-version-2\bearing-digital-twin\backend\main.py" (
        set "PROJECT_DIR=D:\FULL WORK FINAL\bearing-digital-twin-version-2\bearing-digital-twin\"
    )
)

if not defined PROJECT_DIR (
    if exist "D:\NIH_roorkee\FULL WORK FINAL_DIGITAL_NAYA_PANI\bearing-digital-twin-version-2\bearing-digital-twin\backend\main.py" (
        set "PROJECT_DIR=D:\NIH_roorkee\FULL WORK FINAL_DIGITAL_NAYA_PANI\bearing-digital-twin-version-2\bearing-digital-twin\"
    )
)

if not defined PROJECT_DIR (
    echo [ERROR] Could not find the project backend directory!
    echo Please ensure backend\main.py exists.
    pause
    exit /b 1
)

cd /d "!PROJECT_DIR!"
echo [*] Project Location: !PROJECT_DIR!

:: 2. Auto-detect Python executable across all known locations
set "PYTHON_CMD="

if exist "C:\Users\amank\AppData\Local\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=C:\Users\amank\AppData\Local\Programs\Python\Python310\python.exe"
)

if not defined PYTHON_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    )
)

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_CMD=py"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_CMD=.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Could not locate Python automatically.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('"!PYTHON_CMD!" -V 2^>^&1') do set "PY_VER=%%v"
echo [*] Python Detected:  !PY_VER! (!PYTHON_CMD!)

:: 3. Automatically launch default browser to dashboard after 2 seconds
echo [*] Web Dashboard:    http://127.0.0.1:8000
start "" "C:\Windows\System32\cmd.exe" /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: 4. Status banner
echo.
echo ===============================================================================
echo [*] SERVER RUNNING ON http://127.0.0.1:8000
echo ===============================================================================
echo  Default Login Credentials:
echo    - Super Admin:  superadmin  /  SuperAdmin@2026!  (or Suhani@2026)
echo    - Plant Admin:  admin       /  Admin@2026!
echo    - Field Tech:   operator    /  Operator@2026!
echo.
echo  Press [Ctrl + C] anytime in this window to stop the server.
echo ===============================================================================
echo.

:: 5. Launch backend server
"!PYTHON_CMD!" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo [INFO] Server stopped.
pause