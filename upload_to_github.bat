@echo off
setlocal enabledelayedexpansion
title Bearing Fault Detection - GitHub Uploader
color 0B

echo ===============================================================================
echo          BEARING FAULT DETECTION - GITHUB UPLOADER WITH GIT LFS
echo ===============================================================================
echo.

cd /d "%~dp0"
echo [*] Working Directory: %CD%

:: 1. Check Git Installation
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git is not installed or not in PATH!
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

:: 2. Check Git LFS Installation
where git-lfs >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git LFS is not installed!
    echo Please install Git LFS from https://git-lfs.github.com/
    pause
    exit /b 1
)

echo [*] Initializing Git LFS...
git lfs install

:: 3. Initialize Git Repository
if not exist ".git" (
    echo [*] Initializing new Git repository...
    git init
    git branch -M main
) else (
    echo [*] Existing Git repository detected.
)

:: 4. Track Large File Formats with LFS
echo [*] Ensuring large files are tracked by Git LFS...
git lfs track "*.npz"
git lfs track "*.pt"
git lfs track "*.joblib"
git lfs track "*.mat"

:: 5. Stage Files
echo [*] Staging files for commit...
git add .gitattributes
git add .gitignore
git add .

echo.
echo [*] Files ready to commit.
git status -s
echo.

:: 6. Commit
set /p COMMIT_MSG="Enter commit message (or press ENTER for default): "
if "%COMMIT_MSG%"=="" (
    set "COMMIT_MSG=Initial commit: Industrial Bearing Fault Detection Platform"
)

git commit -m "%COMMIT_MSG%"

:: 7. Remote Setup
echo.
echo ===============================================================================
echo                   CONNECT TO YOUR GITHUB REPOSITORY
echo ===============================================================================
echo 1. Go to https://github.com/new and create a new repository (e.g. bearing-fault-detection)
echo    (DO NOT initialize with README, .gitignore, or license - leave it empty)
echo 2. Copy the HTTPS or SSH repository URL.
echo    Example: https://github.com/YOUR_USERNAME/bearing-fault-detection.git
echo ===============================================================================
echo.

set /p REPO_URL="Enter your GitHub Repository URL: "
if "%REPO_URL%"=="" (
    echo [INFO] No URL entered. Skipping push. You can push later using:
    echo        git remote add origin YOUR_URL
    echo        git push -u origin main
    pause
    exit /b 0
)

:: Remove existing origin if set
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%
git branch -M main

:: 8. Push to GitHub
echo.
echo [*] Pushing repository and large LFS files to GitHub...
echo [*] This will upload your code and model weights. Please wait...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ===============================================================================
    echo [SUCCESS] Your project has been uploaded to GitHub successfully!
    echo Repository: %REPO_URL%
    echo ===============================================================================
) else (
    echo.
    echo [NOTICE] If push failed due to authentication:
    echo 1. Make sure you use a GitHub Personal Access Token (PAT) as your password.
    echo    Create one at: https://github.com/settings/tokens
    echo 2. If the remote already has commits, you can run:
    echo    git push -u origin main --force
)

echo.
pause