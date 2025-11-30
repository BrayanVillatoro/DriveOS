@echo off
REM DriveOS Quick Installer
REM Double-click this file to install DriveOS

title DriveOS Installer
color 0A

REM Create log file
set LOGFILE=install_log.txt
echo DriveOS Installation Log - %date% %time% > "%LOGFILE%"
echo ========================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo.
echo ========================================
echo    DriveOS Installation Wizard
echo ========================================
echo.
echo Installation log will be saved to: %LOGFILE%
echo.

REM Check if Python is installed
echo [CHECK] Checking for Python installation... >> "%LOGFILE%"
python --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    color 0C
    echo ERROR: Python is not installed or not in PATH! >> "%LOGFILE%"
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.9, 3.10, or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    echo Check install_log.txt for details.
    echo.
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
echo [STEP 1] Checking Python version... >> "%LOGFILE%"
python -c "import sys; v=sys.version_info; exit(0 if (v.major == 3 and 9 <= v.minor <= 11) else 1)" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    color 0E
    echo WARNING: Python 3.9-3.11 required! >> "%LOGFILE%"
    echo WARNING: Python 3.9-3.11 required!
    echo Your version may not be compatible.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo User chose not to continue >> "%LOGFILE%"
        exit /b 1
    )
    echo User chose to continue anyway >> "%LOGFILE%"
)
echo Step 1 completed successfully >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [2/6] Creating virtual environment...
echo [STEP 2] Creating virtual environment... >> "%LOGFILE%"
if not exist .venv (
    python -m venv .venv >> "%LOGFILE%" 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment >> "%LOGFILE%"
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created >> "%LOGFILE%"
) else (
    echo Virtual environment already exists >> "%LOGFILE%"
)
echo Step 2 completed successfully >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [3/6] Upgrading pip...
echo [STEP 3] Upgrading pip... >> "%LOGFILE%"
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet >> "%LOGFILE%" 2>&1
echo Step 3 completed successfully >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [4/6] Detecting hardware and installing PyTorch...
echo [STEP 4] Detecting hardware and installing PyTorch... >> "%LOGFILE%"
echo.

REM Check for NVIDIA GPU
.venv\Scripts\python.exe -c "import subprocess; result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True); exit(0 if 'NVIDIA' in result.stdout else 1)" >> "%LOGFILE%" 2>&1

if errorlevel 1 (
    echo    No NVIDIA GPU detected - Installing CPU version
    echo No NVIDIA GPU detected - Installing CPU version >> "%LOGFILE%"
    echo    (Training will be slower but compatible with all systems)
    echo.
    .venv\Scripts\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu --quiet >> "%LOGFILE%" 2>&1
) else (
    echo    NVIDIA GPU detected!
    echo NVIDIA GPU detected >> "%LOGFILE%"
    echo.
    echo    Choose PyTorch version:
    echo    1. CUDA 11.8 (Recommended for most NVIDIA GPUs)
    echo    2. CUDA 12.1 (For newest GPUs - RTX 40/50 series)
    echo    3. CPU only (Slower but always compatible)
    echo.
    set /p gpu_choice="Enter choice (1-3) [default: 1]: "
    if "%gpu_choice%"=="" set gpu_choice=1
    echo User choice: %gpu_choice% >> "%LOGFILE%"
    
    if "%gpu_choice%"=="1" (
        echo    Installing PyTorch with CUDA 11.8...
        echo Installing PyTorch with CUDA 11.8 >> "%LOGFILE%"
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet >> "%LOGFILE%" 2>&1
    ) else if "%gpu_choice%"=="2" (
        echo    Installing PyTorch with CUDA 12.1...
        echo Installing PyTorch with CUDA 12.1 >> "%LOGFILE%"
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet >> "%LOGFILE%" 2>&1
    ) else (
        echo    Installing CPU version...
        echo Installing CPU version >> "%LOGFILE%"
        .venv\Scripts\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu --quiet >> "%LOGFILE%" 2>&1
    )
)
echo.
echo Step 4 completed successfully >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [5/6] Installing dependencies...
echo [STEP 5] Installing dependencies... >> "%LOGFILE%"
.venv\Scripts\python.exe -m pip install -r config\requirements.txt --quiet >> "%LOGFILE%" 2>&1
echo Step 5 completed successfully >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [6/6] Creating icon and desktop shortcut...
echo [STEP 6] Creating icon and desktop shortcut... >> "%LOGFILE%"
if not exist launchers\DriveOS.ico (
    echo Icon does not exist, creating... >> "%LOGFILE%"
    .venv\Scripts\python.exe scripts\create_icon.py >> "%LOGFILE%" 2>&1
    if exist launchers\DriveOS.ico (
        echo Icon created successfully >> "%LOGFILE%"
    ) else (
        echo WARNING: Icon creation failed >> "%LOGFILE%"
    )
) else (
    echo Icon already exists >> "%LOGFILE%"
)

echo Creating desktop shortcut... >> "%LOGFILE%"
powershell -ExecutionPolicy Bypass -Command "$desktop = [Environment]::GetFolderPath('Desktop'); $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(Join-Path $desktop 'DriveOS.lnk'); $s.TargetPath = '%CD%\launchers\DriveOS.vbs'; $s.WorkingDirectory = '%CD%'; $s.Description = 'DriveOS Racing Line Analyzer'; $s.IconLocation = '%CD%\launchers\DriveOS.ico,0'; $s.Save(); Write-Output $s.FullName" >> "%LOGFILE%" 2>&1
powershell -ExecutionPolicy Bypass -Command "$desktop = [Environment]::GetFolderPath('Desktop'); Test-Path (Join-Path $desktop 'DriveOS.lnk')" > nul 2>&1
if errorlevel 1 (
    echo    Warning: Could not create shortcut
    echo WARNING: Shortcut creation failed >> "%LOGFILE%"
) else (
    echo    Shortcut created successfully!
    echo Shortcut created successfully >> "%LOGFILE%"
)
echo Step 6 completed >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo DriveOS has been installed successfully!
echo.

REM Display GPU status
echo [VERIFICATION] Checking GPU status... >> "%LOGFILE%"
.venv\Scripts\python.exe -c "import torch; print('GPU Support: ' + ('CUDA ' + torch.version.cuda if torch.cuda.is_available() else 'CPU only')); print('Device: ' + ('NVIDIA GPU detected' if torch.cuda.is_available() else 'CPU'))" 2>> "%LOGFILE%"
if errorlevel 1 (
    echo GPU Support: CPU only
    echo Note: Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads for GPU support
    echo GPU check failed >> "%LOGFILE%"
) else (
    echo GPU check successful >> "%LOGFILE%"
)
echo. >> "%LOGFILE%"

echo.
echo A shortcut has been created on your Desktop.
echo You can also run DriveOS by double-clicking DriveOS.bat
echo.
echo Installation log saved to: %LOGFILE%
echo.
echo Would you like to launch DriveOS now?
set /p launch="Launch now? (y/n): "
echo User launch choice: %launch% >> "%LOGFILE%"
if /i "%launch%"=="y" (
    echo Launching DriveOS... >> "%LOGFILE%"
    start "" "%CD%\launchers\DriveOS.bat"
)

echo.
echo ========================================== >> "%LOGFILE%"
echo Installation completed at %date% %time% >> "%LOGFILE%"
echo ========================================== >> "%LOGFILE%"

echo.
echo Thank you for installing DriveOS!
pause
