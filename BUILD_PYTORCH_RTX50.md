# Building PyTorch with RTX 50 Series Support

## Overview
This guide helps you compile PyTorch from source with sm_120 (RTX 50 series) support.

## ⚠️ Important Notes
- **Time Required:** 2-4 hours
- **Disk Space:** ~30GB
- **RAM:** 16GB+ recommended
- **Difficulty:** Advanced

## Prerequisites

### 1. Visual Studio 2019 or 2022
Download from: https://visualstudio.microsoft.com/downloads/

**During installation, select:**
- "Desktop development with C++"
- Windows 10/11 SDK
- MSVC v142 or v143 build tools

### 2. CUDA Toolkit 12.6+
Your system already has CUDA 13.0 drivers (check with `nvidia-smi`)

Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads

### 3. Git
Download from: https://git-scm.com/download/win

## Build Process

### Option 1: Automated (Recommended)
1. Open **Developer Command Prompt for VS 2022** (search in Start menu)
2. Navigate to DriveOS folder
3. Run: `scripts\build_pytorch_rtx50.bat`
4. Wait 2-4 hours
5. Done!

### Option 2: Manual
```cmd
# Open Developer Command Prompt for VS 2022
cd C:\Users\Braya\OneDrive\Documents\GitHub\DriveOS

# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Set environment variables
set TORCH_CUDA_ARCH_LIST=8.6;9.0;12.0
set USE_CUDA=1
set USE_CUDNN=1
set BUILD_TEST=0
set MAX_JOBS=4

# Install build tools
..\venv\Scripts\python.exe -m pip install cmake ninja

# Build (2-4 hours)
..\venv\Scripts\python.exe setup.py install
```

## What Gets Built
- **sm_86** (Ampere): RTX 30 series
- **sm_90** (Ada Lovelace): RTX 40 series  
- **sm_120** (Blackwell): RTX 50 series ← Your GPU!

## After Building

### Verify Installation
```cmd
.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); x = torch.randn(100,100).cuda(); print('GPU test passed!')"
```

### Test DriveOS
1. Launch DriveOS
2. Go to "Analyze Video" tab
3. Select "GPU - CUDA" mode
4. Process a video
5. Should now show ~10-20x faster speed!

## Troubleshooting

### Build Fails with "Out of Memory"
Reduce parallel jobs:
```cmd
set MAX_JOBS=2
```

### "cl.exe not found"
You need to run from **Developer Command Prompt for VS**, not regular Command Prompt.

### "nvcc not found"  
Add CUDA to PATH:
```cmd
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
```

### Build takes too long
This is normal! Building PyTorch from source takes 2-4 hours even on high-end systems.

## Alternative: Wait for Official Build
PyTorch 2.11+ (expected Q1 2025) will have RTX 50 support out of the box.

To update when available:
```cmd
.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio
```

## Current Status
✅ DriveOS works on CPU mode (slower)  
⏳ Building from source enables GPU mode (10-20x faster)  
🔜 Official PyTorch 2.11+ will support RTX 50 natively
