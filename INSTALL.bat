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

REM Check if Python is installed - prefer Python 3.11 for best compatibility
echo [CHECK] Checking for Python installation... >> "%LOGFILE%"
set PYTHON_CMD=python
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    echo Found Python 3.11 - using for best compatibility >> "%LOGFILE%"
) else (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.12
        echo Found Python 3.12 >> "%LOGFILE%"
    ) else (
        python --version >> "%LOGFILE%" 2>&1
        if errorlevel 1 (
            color 0C
            echo ERROR: Python is not installed or not in PATH! >> "%LOGFILE%"
            echo ERROR: Python is not installed or not in PATH!
            echo.
            echo Please install Python 3.9-3.12 from:
            echo https://www.python.org/downloads/
            echo.
            echo Make sure to check "Add Python to PATH" during installation!
            echo.
            echo Check install_log.txt for details.
            echo.
            pause
            exit /b 1
        )
    )
)
%PYTHON_CMD% --version >> "%LOGFILE%"

echo [1/6] Checking Python version...
echo [STEP 1] Checking Python version... >> "%LOGFILE%"
%PYTHON_CMD% -c "import sys; v=sys.version_info; exit(0 if (v.major == 3 and 9 <= v.minor <= 12) else 1)" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    color 0E
    echo WARNING: Python 3.9-3.12 required! >> "%LOGFILE%"
    echo WARNING: Python 3.9-3.12 required!
    echo Your version may not be compatible with PyTorch.
    %PYTHON_CMD% -c "import sys; print(f'Detected: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
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
    %PYTHON_CMD% -m venv .venv >> "%LOGFILE%" 2>&1
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

REM Check for NVIDIA GPU using PowerShell (works on all Windows versions)
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "NVIDIA" > nul 2>&1

if errorlevel 1 (
    echo    No NVIDIA GPU detected - Installing CPU version
    echo No NVIDIA GPU detected - Installing CPU version >> "%LOGFILE%"
    echo    (Training will be slower but compatible with all systems)
    echo.
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet >> "%LOGFILE%" 2>&1
) else (
    echo    NVIDIA GPU detected!
    echo NVIDIA GPU detected >> "%LOGFILE%"
    echo.
    echo    Choose PyTorch version:
    echo    1. CUDA 11.8 (Recommended for GTX/RTX 20/30 series)
    echo    2. CUDA 12.1 (For RTX 40 series)
    echo    3. PyTorch Nightly with CUDA 12.6 (For RTX 50 series - experimental, uses CPU fallback)
    echo    4. CPU only (Slower but always compatible)
    echo.
    set /p gpu_choice="Enter choice (1-4) [default: 1]: "
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
    ) else if "%gpu_choice%"=="3" (
        echo    Installing PyTorch Nightly with CUDA 12.6 ^(RTX 50 series^)...
        echo    WARNING: RTX 50 series not fully supported yet - will use CPU fallback
        echo Installing PyTorch Nightly CUDA 12.6 >> "%LOGFILE%"
        echo    Note: This is a large download and may take 5-10 minutes...
        .venv\Scripts\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 >> "%LOGFILE%" 2>&1
        if errorlevel 1 (
            echo    Nightly build failed, trying stable CUDA 12.1...
            echo Nightly build failed, falling back to CUDA 12.1 stable >> "%LOGFILE%"
            .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 >> "%LOGFILE%" 2>&1
        )
    ) else (
        echo    Installing CPU version...
        echo Installing CPU version >> "%LOGFILE%"
        .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet >> "%LOGFILE%" 2>&1
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
powershell -ExecutionPolicy Bypass -Command "$desktop = [Environment]::GetFolderPath('Desktop'); $shortcutPath = Join-Path $desktop 'DriveOS.lnk'; $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut($shortcutPath); $s.TargetPath = '%CD%\launchers\DriveOS.vbs'; $s.WorkingDirectory = '%CD%'; $s.Description = 'DriveOS Racing Line Analyzer'; $s.IconLocation = '%CD%\launchers\DriveOS.ico,0'; $s.Save(); Write-Output ('Shortcut created at: ' + $s.FullName)" >> "%LOGFILE%" 2>&1
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
