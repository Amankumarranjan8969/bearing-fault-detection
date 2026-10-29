@echo off
setlocal enabledelayedexpansion
title Bearing Fault Detection - Enterprise Condition Monitoring
color 0B

echo ===============================================================================
echo          BEARING FAULT DETECTION - ENTERPRISE CONDITION MONITORING
echo               Physics-Informed AI - RAG Diagnostics - NASA IMS
echo ===============================================================================
echo.

:: 1. Navigate to script directory
cd /d "%~dp0"
echo [*] Working Directory: %CD%

:: 2. Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -V') do set PYTHON_VERSION=%%i
echo [*] Detected: %PYTHON_VERSION%

:: 3. Check for main project backend
if not exist "backend\main.py" (
    echo [ERROR] backend\main.py not found in current directory!
    echo Current folder: %CD%
    echo.
    pause
    exit /b 1
)

:: 4. Launch web browser automatically after a short delay
echo [*] Launching browser to http://localhost:8000 ...
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

:: 5. Start Uvicorn Server
echo.
echo [OK] Starting Bearing Fault Detection backend server on port 8000...
echo [OK] Default Logins:
echo      - Super Admin:  superadmin  /  SuperAdmin@2026!  (Full Master Control)
echo      - Plant Admin:  admin       /  Admin@2026!       (Plant Operations)
echo      - Operator:     operator    /  Operator@2026!    (Field Monitoring)
echo.
echo Press [Ctrl + C] in this window anytime to stop the server.
echo ===============================================================================
echo.

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo [INFO] Bearing Fault Detection Server has stopped.
pause
