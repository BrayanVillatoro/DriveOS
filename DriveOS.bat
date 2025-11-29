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

REM Check for virtual environment and launch without console window
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" launch_gui.pyw
) else if exist ".venv311\Scripts\pythonw.exe" (
    start "" ".venv311\Scripts\pythonw.exe" launch_gui.pyw
) else (
    echo Error: Python environment not found!
    echo Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

REM Exit immediately after launching (don't keep console open)
exit
