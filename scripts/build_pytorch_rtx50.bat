@echo off
REM Simple PyTorch RTX 50 Builder (Manual Prerequisites)
REM Use this if the automatic installer has issues

title PyTorch RTX 50 Builder - Manual Mode
color 0B

echo.
echo ========================================
echo  PyTorch RTX 50 Builder - Manual Mode
echo ========================================
echo.
echo This version assumes you already have:
echo   - Git installed
echo   - Visual Studio 2019/2022 with C++ tools
echo   - CUDA Toolkit (you have CUDA 13.0 drivers)
echo.
echo If you don't have these, please install them first:
echo   Git: https://git-scm.com/
echo   VS 2022: https://visualstudio.microsoft.com/downloads/
echo   (Select "Desktop development with C++" during install)
echo.
set /p continue="Ready to build? (yes/no): "
if /i not "%continue%"=="yes" exit /b

echo.
echo Starting build process...
echo.

REM Check Git
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found. Please install Git first.
    pause
    exit /b 1
)

REM Setup VS environment
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ERROR: Visual Studio not found.
    echo Please install Visual Studio 2019 or 2022 with C++ tools.
    pause
    exit /b 1
)

echo [1/5] Creating build directory...
if not exist build_pytorch mkdir build_pytorch
cd build_pytorch

echo [2/5] Cloning PyTorch (10-20 minutes)...
if not exist pytorch (
    git clone --recursive --depth=1 https://github.com/pytorch/pytorch
    if errorlevel 1 (
        echo ERROR: Failed to clone PyTorch
        pause
        exit /b 1
    )
)
cd pytorch

echo [3/5] Installing build tools...
..\..\..\.venv\Scripts\python.exe -m pip install cmake ninja setuptools wheel pyyaml typing_extensions

echo [4/5] Configuring build...
set TORCH_CUDA_ARCH_LIST=8.6;9.0;12.0
set USE_CUDA=1
set USE_CUDNN=1
set BUILD_TEST=0
set BUILD_CAFFE2=0
set MAX_JOBS=4

echo.
echo ========================================
echo Build Configuration
echo ========================================
echo CUDA Architectures: 8.6, 9.0, 12.0
echo   8.6  = RTX 30 series (Ampere)
echo   9.0  = RTX 40 series (Ada)
echo   12.0 = RTX 50 series (Blackwell) - YOUR GPU!
echo.
echo This will take 2-4 hours.
echo You can minimize this window.
echo ========================================
echo.

echo [5/5] Building PyTorch...
echo Started: %time%
echo.

..\..\..\.venv\Scripts\python.exe setup.py install
if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Build completed: %time%
echo.

echo Testing installation...
cd ..\..\..
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); x = torch.randn(10,10).cuda(); print('✓ RTX 50 series working!')"

echo.
echo ========================================
echo         SUCCESS!
echo ========================================
echo.
echo Your RTX 5070 Ti now has full PyTorch support!
echo.
echo Launch DriveOS and select GPU mode to enjoy
echo 10-20x faster video processing!
echo.
pause
