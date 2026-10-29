<#
.SYNOPSIS
    One-Click PowerShell Launcher for Enterprise Bearing Fault Detection.
.DESCRIPTION
    Checks system environment, starts FastAPI backend, and automatically opens the browser.
#>

$Host.UI.RawUI.WindowTitle = "Bearing Fault Detection · Enterprise Platform"

Write-Host @"
===============================================================================
          BEARING FAULT DETECTION · ENTERPRISE CONDITION MONITORING
              Physics-Informed AI · RAG Diagnostics · NASA IMS
===============================================================================
"@ -ForegroundColor Cyan

# Ensure script root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "[*] Platform Directory: $ScriptDir" -ForegroundColor Gray

# Check Python
try {
    $pythonVer = python --version 2>&1
    Write-Host "[*] Python Environment: $pythonVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python was not found in PATH! Please install Python 3.10+." -ForegroundColor Red
    pause
    exit 1
}

# Verify entrypoint
if (-not (Test-Path "backend\main.py")) {
    Write-Host "[ERROR] backend\main.py not found in $ScriptDir" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "`n[*] Starting Bearing Fault Detection Engine on http://localhost:8000..." -ForegroundColor Yellow
Write-Host "[*] Pre-Configured Credentials:" -ForegroundColor Cyan
Write-Host "    - Super Admin:  superadmin  /  SuperAdmin@2026!  (Master Control)" -ForegroundColor White
Write-Host "    - Plant Admin:  admin       /  Admin@2026!       (Plant Operations)" -ForegroundColor White
Write-Host "    - Operator:     operator    /  Operator@2026!    (Field Monitoring)" -ForegroundColor White

# Launch browser asynchronously after 2 seconds
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:8000"
} | Out-Null

Write-Host "`n[OK] Opening dashboard in your default browser..." -ForegroundColor Green
Write-Host "[OK] Press [Ctrl + C] anytime to stop the server.`n" -ForegroundColor Gray

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
