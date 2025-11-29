@echo off
REM DriveOS Quick Installer
REM Double-click this file to install DriveOS

title DriveOS Installer
color 0A

echo.
echo ========================================
echo    DriveOS Installation Wizard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
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

echo [1/6] Checking Python version...
python -c "import sys; v=sys.version_info; exit(0 if 3.9<=v.major+v.minor/10<3.12 else 1)"
if errorlevel 1 (
    color 0E
    echo WARNING: Python 3.9-3.11 required!
    echo Your version may not be compatible.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo [2/6] Creating virtual environment...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo [3/6] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [4/6] Detecting hardware and installing PyTorch...
echo.

REM Check for NVIDIA GPU
.venv\Scripts\python.exe -c "import subprocess; result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True); exit(0 if 'NVIDIA' in result.stdout else 1)" >nul 2>&1

if errorlevel 1 (
    echo    No NVIDIA GPU detected - Installing CPU version
    echo    (Training will be slower but compatible with all systems)
    echo.
    .venv\Scripts\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo    NVIDIA GPU detected!
    echo.
    echo    Choose PyTorch version:
    echo    1. CUDA 11.8 (Recommended for most NVIDIA GPUs)
    echo    2. CUDA 12.1 (For newest GPUs - RTX 40/50 series)
    echo    3. CPU only (Slower but always compatible)
    echo.
    set /p gpu_choice="Enter choice (1-3) [default: 1]: "
    if "%gpu_choice%"=="" set gpu_choice=1
    
    if "%gpu_choice%"=="1" (
        echo    Installing PyTorch with CUDA 11.8...
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
    ) else if "%gpu_choice%"=="2" (
        echo    Installing PyTorch with CUDA 12.1...
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    ) else (
        echo    Installing CPU version...
        .venv\Scripts\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu --quiet
    )
)
echo.

echo [5/6] Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet

echo [6/6] Creating icon and desktop shortcut...
if not exist DriveOS.ico (
    .venv\Scripts\python.exe create_icon.py >nul 2>&1
)

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = "%USERPROFILE%\Desktop\DriveOS.lnk" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "%CD%\DriveOS.bat" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%CD%" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "DriveOS Racing Line Analyzer" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.IconLocation = "%CD%\DriveOS.ico" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"
cscript //nologo "%TEMP%\CreateShortcut.vbs" 2>nul
if exist "%USERPROFILE%\Desktop\DriveOS.lnk" (
    echo    Shortcut created successfully!
) else (
    echo    Warning: Could not create shortcut
)
del "%TEMP%\CreateShortcut.vbs" 2>nul

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo DriveOS has been installed successfully!
echo.

REM Display GPU status
.venv\Scripts\python.exe -c "import torch; print('GPU Support: ' + ('CUDA ' + torch.version.cuda if torch.cuda.is_available() else 'CPU only')); print('Device: ' + ('NVIDIA GPU detected' if torch.cuda.is_available() else 'CPU'))" 2>nul
if errorlevel 1 (
    echo GPU Support: CPU only
    echo Note: Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads for GPU support
)

echo.
echo A shortcut has been created on your Desktop.
echo You can also run DriveOS by double-clicking DriveOS.bat
echo.
echo Would you like to launch DriveOS now?
set /p launch="Launch now? (y/n): "
if /i "%launch%"=="y" (
    start "" "%CD%\DriveOS.bat"
)

echo.
echo Thank you for installing DriveOS!
pause
