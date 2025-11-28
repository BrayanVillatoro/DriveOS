@echo off
REM DriveOS Quick Installer
REM Double-click this file to install DriveOS

echo.
echo ========================================
echo    DriveOS Installation Wizard
echo ========================================
echo.
echo Starting installer...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.9, 3.10, or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

REM Install installer dependencies if needed
echo Checking installer requirements...
python -c "import winshell" >nul 2>&1
if errorlevel 1 (
    echo Installing installer dependencies...
    python -m pip install --quiet pywin32 winshell
)

REM Run the installer
python installer.py

if errorlevel 1 (
    echo.
    echo Installation failed or was cancelled.
    pause
    exit /b 1
)

echo.
echo Installation complete!
pause
