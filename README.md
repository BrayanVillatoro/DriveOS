# 🏁 DriveOS - AI Racing Line Analyzer

**AI-powered racing line analysis to find the fastest, most efficient path around any track**

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

## 🚀 Quick Start - Just 3 Steps!

### 📥 Installation (2-5 minutes)

1. **Download** this repository
   - Click the green "Code" button → "Download ZIP"
   - Or clone: `git clone https://github.com/BrayanVillatoro/DriveOS.git`

2. **Extract** the ZIP file to any location on your computer

3. **Double-click `INSTALL.bat`** and let it do everything!

That's it! The installer will automatically:
- ✅ Detect your hardware (NVIDIA GPU, CPU, etc.)
- ✅ Install the correct PyTorch version (CUDA for NVIDIA GPUs)
- ✅ Install all dependencies (OpenCV, NumPy, etc.)
- ✅ Create a virtual environment
- ✅ Create a Desktop shortcut
- ✅ Confirm GPU support status

After installation completes, you'll find a **DriveOS shortcut on your Desktop**. Just double-click it to launch!

### System Requirements

**Minimum Requirements:**
- **Operating System:** Windows 10 or newer (64-bit)
- **Python:** 3.9, 3.10, or 3.11 ([Download here](https://www.python.org/downloads/))
  - ⚠️ **IMPORTANT:** Check "Add Python to PATH" during Python installation
- **Processor:** Intel Core i5 / AMD Ryzen 5 or better (4+ cores)
- **RAM:** 8 GB minimum
- **Disk Space:** 5 GB free space (for installation and models)
- **Graphics:** Any GPU or integrated graphics (for display only)

**Recommended for Better Performance:**
- **Processor:** Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores, 16+ threads)
- **RAM:** 16 GB or more
- **Disk Space:** 10 GB+ (for training data storage)

**NVIDIA GPU Acceleration (Optional but Recommended for Training):**
- **Graphics Card:** NVIDIA GPU with CUDA support
  - GTX 1060 6GB or better (older GPUs)
  - RTX 2060 or better (recommended)
  - RTX 3060 or better (great performance)
  - RTX 4060 or better (excellent performance)
  - RTX 5070 Ti / 5090 (requires PyTorch 2.9+ - coming soon)
- **CUDA Compute Capability:** 3.5 or higher
- **VRAM:** 4 GB minimum, 6 GB+ recommended
- **Driver:** Latest NVIDIA Game Ready or Studio Drivers

> **Note:** The installer automatically detects NVIDIA GPUs and offers CUDA installation. GPU acceleration speeds up training by 10-20x but is not required - CPU training works fine, just slower.

**Why NVIDIA?**
- **Training Speed:** GPU training is 10-20x faster than CPU
- **Real-time Processing:** Better frame rates during live analysis
- **Larger Models:** Can train with bigger batch sizes
- **Native Support:** PyTorch has excellent CUDA optimization

## ✨ Features

- **🎥 Multiple Input Sources:**
  - Video files (MP4, AVI, MOV, MKV)
  - Webcam/Camera feed
  - Screen capture (perfect for sim racing!)

- **🤖 AI-Powered Analysis:**
  - Deep learning models (DeepLabV3 + LSTM)
  - Real-time racing line detection
  - Track edge and curb identification
  - Off-track area detection

- **📊 Visual Feedback:**
  - Purple line = Optimal racing line
  - Cyan overlay = Track edges
  - Green zones = Safe racing area
  - Red highlights = Off-track areas

- **🎯 Professional GUI:**
  - Easy-to-use interface
  - Real-time statistics
  - Progress tracking
  - Batch video processing

## 🎮 Perfect for Sim Racing

DriveOS supports **screen capture**, making it perfect for analyzing your sim racing sessions in real-time! Works with:
- iRacing
- Assetto Corsa / Assetto Corsa Competizione
- F1 games
- Gran Turismo (via capture card)
- Any racing game or simulator

## 📖 How to Use

### 1. Analyze a Video

1. Launch DriveOS
2. Go to the **"Analyze Video"** tab
3. Click **"Select Video"** and choose your racing video
4. Click **"ANALYZE VIDEO WITH AI"**
5. Wait for processing to complete
6. Your analyzed video will be saved with the optimal racing line highlighted!

### 2. Live Preview

1. Go to the **"Live Preview"** tab
2. Choose your source:
   - **Video File:** Select a video to preview
   - **Webcam/Camera:** Select camera index and test connection
   - **Screen Capture:** Capture your sim racing gameplay in real-time
3. Click **"Start Processing"**
4. Watch real-time racing line analysis!

### 3. Train the Model

1. Go to the **"Train Model"** tab
2. Click **"Generate Training Data"** and select a racing video
3. Adjust training parameters (epochs, batch size, learning rate)
4. Click **"Start Training"**
5. Wait for training to complete (may take 30-60 minutes)

## 🛠️ Manual Installation (For Developers)

If you want to modify the code or contribute to the project:

```bash
# Clone the repository
git clone https://github.com/BrayanVillatoro/DriveOS.git
cd DriveOS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows

# Install PyTorch with CUDA 11.8 (for NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR install CPU-only version (slower but works on any PC)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Launch the GUI
python launch_gui.py
```

**Choosing the Right PyTorch:**
- **CUDA 11.8:** Best for GTX 1000/RTX 2000/3000 series
- **CUDA 12.1:** Better for RTX 4000 series
- **CPU:** Works everywhere but ~10x slower for training

## 🧠 Technical Details

### Architecture

- **Vision Model:** DeepLabV3 with ResNet50 backbone (pretrained on COCO)
- **Telemetry Model:** LSTM networks for temporal sequence analysis
- **Fusion Model:** Combines vision and telemetry predictions
- **Training:** Supervised learning with auto-generated labels from track detection
- **Inference Engine:** Batch processing with multi-threaded CPU or GPU acceleration

### Performance

**CPU Mode (All PCs):**
- ~5-10 FPS processing speed
- ~100-200ms per frame
- Uses 8-16 threads for parallel processing
- Good for post-session analysis

**GPU Mode (NVIDIA GPUs):**
- ~30-60 FPS processing speed
- ~15-30ms per frame
- 10-20x faster training
- Excellent for real-time analysis

**Processing:**
- **Input Resolution:** 1920x1080 or 1280x720
- **Model Resolution:** 320x320 (optimized for speed)
- **Batch Size:** Adjustable (2-16 depending on VRAM/RAM)
- **Output:** Same resolution as input with overlays

### Files & Directories

```
DriveOS/
├── INSTALL.bat          # Quick installer launcher
├── installer.py         # Installation wizard
├── launch_gui.py        # GUI launcher
├── DriveOS.bat         # Direct launcher (after installation)
├── src/
│   ├── gui.py          # Main GUI application
│   ├── inference.py    # Inference engine
│   ├── models.py       # ML model architectures
│   ├── train.py        # Training script
│   └── config.py       # Configuration
├── models/             # Trained model weights
├── data/
│   └── training/       # Training data
└── requirements.txt    # Python dependencies
```

## 🎯 GPU vs CPU: What to Expect

**CPU Training (No NVIDIA GPU):**
- Training 20 epochs: ~30-60 minutes
- Processing 1 hour of video: ~10-20 minutes
- Live preview: 5-10 FPS
- ✅ Works on any PC
- ✅ No special hardware needed

**GPU Training (NVIDIA GPU with CUDA):**
- Training 20 epochs: ~3-5 minutes (10-20x faster)
- Processing 1 hour of video: ~1-2 minutes
- Live preview: 30-60 FPS
- ✅ Much faster training
- ✅ Real-time analysis possible
- ⚠️ Requires NVIDIA GPU (GTX 1060+)

**Recommendation:** Start with CPU mode to learn the software. If you plan to train frequently or need real-time analysis, NVIDIA GPU is highly recommended.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is research software.** DriveOS is provided for educational and research purposes. The racing line suggestions are AI-generated and should not be considered as professional racing advice. Always prioritize safety when racing.

## 🐛 Troubleshooting

### "Python not found" error
- Install Python 3.9-3.11 from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### Installation fails
- Run `INSTALL.bat` as Administrator
- Make sure you have internet connection
- Check that you have ~2 GB free disk space

### Screen capture not working
- Install screen capture support: The installer will prompt you
- Or manually: `pip install mss`

### Slow processing
- Processing speed depends on your CPU
- Close other applications during processing
- Consider upgrading to a multi-core CPU (16+ threads recommended)

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/BrayanVillatoro/DriveOS/issues)
- Check existing issues for solutions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ for the racing community

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework and CUDA support
- **OpenCV** - Computer vision and video processing
- **NVIDIA** - CUDA and GPU acceleration technology
- **DeepLab** - Semantic segmentation architecture

## 🗺️ Roadmap

- [ ] Pre-trained models for common tracks
- [ ] Multi-car comparison and analysis
- [ ] Track database with optimal lines
- [ ] Enhanced real-time processing
- [ ] Mobile companion app
- [ ] Cloud-based training
- [ ] Advanced telemetry integration
- [ ] Multi-session comparison tools

---

**DriveOS** - Drive faster, smarter, better. 🏁
