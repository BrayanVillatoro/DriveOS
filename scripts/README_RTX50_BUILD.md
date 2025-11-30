# 🚀 Build PyTorch with RTX 50 Series Support

## Prerequisites (Install These First)

### 1. Visual Studio 2022 (Required)
Download and install from: https://visualstudio.microsoft.com/downloads/

**Important:** During installation, select:
- ✅ "Desktop development with C++"
- ✅ Windows 10/11 SDK

This is the C++ compiler needed to build PyTorch.

### 2. Git (Probably Already Have)
You likely already have this. To check, open PowerShell and type: `git --version`

If not installed: https://git-scm.com/

## Building PyTorch

### Step 1: Run the Script
Once Visual Studio is installed:
1. Go to `scripts` folder
2. **Double-click** `build_pytorch_rtx50.bat`
3. Type **"yes"** and press Enter
4. Wait 2-4 hours

### Step 2: Walk Away
- Takes 2-4 hours to compile
- You can minimize the window
- Your computer will be slow during this time
- Check back when it says "SUCCESS!"

### Step 3: Done!
- Launch DriveOS
- Select "GPU - CUDA" mode
- Enjoy 10-20x faster processing!

## What It Does

✅ Downloads PyTorch source code (5GB)  
✅ Compiles CUDA kernels for RTX 50 (sm_120)  
✅ Installs into your DriveOS virtual environment  
✅ Tests that everything works  

## Requirements

- Visual Studio 2022 with C++ tools (see above)
- 30GB free disk space
- Good internet connection
- 2-4 hours of patience

## Troubleshooting

### "Visual Studio not found"
→ Install Visual Studio 2022 with "Desktop development with C++" first

### "Git not found"
→ Install Git from https://git-scm.com/

### "Out of memory" during build
→ Close Chrome, games, and other programs

### Build fails
→ Run the script again, it will resume from where it stopped

### Takes too long
→ This is normal! Compiling PyTorch from source takes 2-4 hours

## After Building

Want to free up 30GB of disk space?

Delete the `build_pytorch` folder - your custom PyTorch is already installed in DriveOS!

## Alternative: Just Wait

PyTorch 2.11 (expected Q1 2025) will support RTX 50 series out of the box.

Then you can simply run:
```
pip install --upgrade torch
```

CPU mode works fine until then!
