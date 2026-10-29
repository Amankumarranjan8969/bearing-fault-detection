@echo off
setlocal enabledelayedexpansion
title Bearing Fault Detection - Enterprise Condition Monitoring
color 0B

echo ===============================================================================
echo          BEARING FAULT DETECTION - ENTERPRISE CONDITION MONITORING
echo               Physics-Informed AI - RAG Diagnostics - NASA IMS
echo ===============================================================================
echo.

cd /d "%~dp0"
call launch_bearing_twin.bat
