@echo off
setlocal enabledelayedexpansion
title Bearing Fault Detection & Digital Twin - Enterprise Platform
color 0A

echo ===============================================================================
echo        BEARING FAULT DETECTION & DIGITAL TWIN - ENTERPRISE PLATFORM
echo               Physics-Informed AI - RAG Diagnostics - 3D Twin
echo ===============================================================================
echo.

:: 1. Auto-detect project directory
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
    if exist "C:\Users\amank\Downloads\FULL WORK FINAL\bearing-digital-twin-version-2\bearing-digital-twin\backend\main.py" (
        set "PROJECT_DIR=C:\Users\amank\Downloads\FULL WORK FINAL\bearing-digital-twin-version-2\bearing-digital-twin\"
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

:: 2. Find Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in system PATH.
    echo Please install Python 3.10+ from python.org and add to PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -V') do set "PY_VER=%%i"
echo [*] Python Version:   !PY_VER!

:: 3. Launch Browser in background with 2s delay
echo [*] Web Dashboard:    http://127.0.0.1:8000
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: 4. Information Banner
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

:: 5. Run Uvicorn Server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo [INFO] Server stopped.
pause