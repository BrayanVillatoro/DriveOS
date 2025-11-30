@echo off
REM Quick launcher for DriveOS
REM This is a convenience launcher - you can also use the desktop shortcut

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\pythonw.exe" (
    echo Error: Virtual environment not found!
    echo Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

REM Launch without console window
start "" ".venv\Scripts\pythonw.exe" "launchers\launch_gui.pyw"
