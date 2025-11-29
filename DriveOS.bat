@echo off
title DriveOS - Racing Line Analyzer
echo.
echo ================================================
echo          DriveOS Racing Line Analyzer
echo ================================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"

REM Check for virtual environment
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch_gui.py
) else if exist ".venv311\Scripts\python.exe" (
    ".venv311\Scripts\python.exe" launch_gui.py
) else (
    echo Error: Python environment not found!
    echo Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)
