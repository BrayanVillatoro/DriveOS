# 🏁 DriveOS - AI Racing Line Analyzer

**AI-powered racing line analysis to find the fastest, most efficient path around any track**

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

## 🤖 What is DriveOS?

DriveOS is a **deep learning application** that trains neural networks to detect racing track boundaries in real-time video footage. Inspired by comma.ai's openpilot lane detection system, DriveOS uses advanced computer vision to understand track geometry from a driver's perspective.

### Architecture: Openpilot-Inspired Edge Detection

**Dual-Head Neural Network**

1. **Track Segmentation Head** - U-Net architecture for drivable surface detection
   - Encoder-decoder with skip connections for precise boundaries
   - Outputs pixel-wise classification (track vs. non-track)
   - Handles lighting variations and track surface changes

2. **Edge Regression Head** - Direct coordinate prediction (like openpilot)
   - Predicts left/right track edges as **3D coordinates in world space**
   - **33 points per edge** with quadratic spacing (dense near, sparse far)
   - X-coordinate: Distance ahead (0-50 meters)
   - Y-coordinate: Lateral offset from center (meters)
   - Mixture Density Network outputs with uncertainty estimation

**Why This Approach Works:**

- **3D World Coordinates** - Unlike pixel-based methods that bunch at image horizon, world-space coordinates naturally spread points based on actual track distance
- **Explicit Supervision** - Model learns exact edge positions rather than deriving them from segmentation masks
- **Temporal Smoothing** - Exponential moving average (EMA) with α=0.7 for stable, jitter-free edges
- **Uncertainty Aware** - Confidence scores allow filtering of unreliable predictions

**Training Process:**

1. **Data Annotation** - Built-in tool to label track boundaries
2. **Edge Label Generation** - Automatically converts segmentation masks to 3D edge coordinates
3. **Multi-Task Learning** - Trains both segmentation and edge regression simultaneously
4. **Edge Coordinate Loss** - L1 loss on (X,Y) positions + confidence classification
5. **Inference** - Real-time prediction with temporal smoothing for production-quality results

**Key Innovation:** Following openpilot's proven approach of predicting explicit coordinates in world space rather than pixel space eliminates the "horizon bunching" problem common in image-based lane detection

## 🚀 Quick Start

1. **Download** this repository (Code → Download ZIP)
2. **Extract** to any location
3. **Double-click `INSTALL.bat`**

The installer automatically handles everything - Python environment, dependencies, GPU detection, and creates a desktop shortcut. Launch DriveOS from your desktop after installation!

### System Requirements

**Minimum:**
- Windows 10+ (64-bit)
- Python 3.9-3.11 ([Download](https://www.python.org/downloads/)) - **Check "Add Python to PATH"**
- 8 GB RAM, 5 GB free space
- Intel Core i5 / AMD Ryzen 5 (4+ cores)

**Recommended:**
- 16 GB RAM, 10 GB free space
- Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores)
- NVIDIA GPU (GTX 1060 6GB+, RTX 2060+) - 10-20x faster training

## ✨ Features

- **🎥 Multiple Input Sources:**
  - Video files (MP4, AVI, MOV, MKV)
  - Webcam/Camera feed
  - Screen capture (perfect for sim racing!)

- **🤖 AI-Powered Analysis:**
  - Dual-head architecture: U-Net segmentation + Edge regression
  - Openpilot-inspired 3D coordinate prediction (world-space, not pixel-space)
  - 33-point edge detection with quadratic spacing (0-50m ahead)
  - Mixture Density Network with uncertainty estimation
  - Temporal smoothing (EMA α=0.7) for stable predictions

- **📊 Visual Feedback:**
  - Green overlay = Detected track surface
  - White polylines = Left/right track edges in world coordinates
  - Edge confidence scores (L/R) displayed
  - Real-time inference time monitoring
  - Smooth, jitter-free edge tracking

- **🎯 Professional GUI:**
  - Easy to use interface
  - Real time statistics
  - Progress tracking
  - Batch video processing

## 📸 Screenshots

### Analyze Video
Process racing videos and overlay the optimal racing line.

![Analyze Video](readme_images/analyze-video.png)

### Live Preview
Real time racing line analysis from video files, webcams, or screen capture.

![Live Preview](readme_images/live%20view.png)

### Create Training Data
Interactive annotation tool for creating custom training datasets.

![Training Data Tool](readme_images/how-to-train.png)

### Train Your Model
Train custom AI models on your own racing footage.

![Model Training](readme_images/training-model.png)

## 🎮 Perfect for Sim Racing

DriveOS supports **screen capture**, making it perfect for analyzing your sim racing sessions in real time! Works with:
- iRacing
- Assetto Corsa / Assetto Corsa Competizione
- F1 games
- Gran Turismo (via capture card)
- Any racing game or simulator

## 📖 How to Use

### Analyze Video
1. Launch DriveOS → **"Analyze Video"** tab
2. Select your racing video
3. Click **"ANALYZE VIDEO WITH AI"**
4. Get processed video with racing line overlay

### Live Preview
1. **"Live Preview"** tab → Choose source (Video, Webcam, or Screen Capture)
2. Click **"Start Processing"** for real time analysis

### Create Training Data
1. **"Create Training Data"** tab → Select video → **"Launch Annotation Tool"**
2. Draw track boundaries: left edge (red), right edge (blue)
3. Press **SPACE** to save frame, **Q** when done
4. Annotate 50-100 diverse frames for best results

### Train Custom Model
1. **"Train Model"** tab → Select training data directory
2. Set environment variables (or use defaults):
   - `USE_EDGE_HEAD=true` - Enable openpilot-style edge regression
   - `EDGE_LOSS_WEIGHT=1.0` - Weight for edge coordinate loss
   - `EDGE_CONF_WEIGHT=0.5` - Weight for edge confidence loss
3. Adjust parameters (epochs: 50-100, batch size: 4-8)
4. Click **"Start Training"** (5-10 min GPU, 30-60 min CPU)
5. Edge labels auto-generated from segmentation masks before training

## 🛠️ Manual Installation (Developers)

```bash
git clone https://github.com/BrayanVillatoro/DriveOS.git
cd DriveOS
python -m venv .venv
.venv\Scripts\activate

# GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU: pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r config/requirements.txt
python launchers/launch_gui.py
```

## 🧠 Technical Details

**Architecture:** Dual-head U-Net inspired by comma.ai's openpilot vision model

**Segmentation Head:**
- **Encoder:** 5 downsampling blocks (64→1024 filters)
- **Decoder:** 4 upsampling blocks with skip connections
- **Output:** Track surface segmentation (2 classes)

**Edge Regression Head (Openpilot-Style):**
- **Input:** Multi-scale features from U-Net encoder
- **Output:** 2 edges × 33 points × 2 coordinates (X, Y)
- **Coordinate System:**
  - X: Distance ahead (0-50m, quadratic spacing)
  - Y: Lateral offset from center (meters)
- **Uncertainty:** Per-point std deviations + edge confidence scores
- **Loss:** L1 on coordinates + BCE on confidence

**Training:**
- **Multi-task loss:** Segmentation (BCE) + Edge coordinates (L1 + confidence)
- **Loss weights:** Seg=1.0, Edge=1.0, Confidence=0.5
- **Data:** Auto-generated edge labels from segmentation masks
- **Optimizer:** Adam with learning rate scheduling

**Inference Pipeline:**
1. Forward pass through dual-head network
2. Temporal smoothing: EMA (α=0.7) on edge coordinates
3. Convert world coordinates to pixel space for visualization
4. Render track overlay + edge polylines

**Performance:**
- **CPU:** ~175-200ms/frame (5 FPS)
- **GPU:** ~15-30ms/frame (30-60 FPS)
- **Model size:** 320×320 input → any output resolution

**Key Advantages:**
- World-space coordinates eliminate horizon bunching
- Explicit edge supervision improves boundary accuracy
- Temporal smoothing provides production-quality stability
- Uncertainty-aware predictions for reliability filtering

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is research software.** DriveOS is provided for educational and research purposes. The racing line suggestions are AI generated and should not be considered as professional racing advice. Always prioritize safety when racing.

## 🐛 Troubleshooting

- **"Python not found":** Install Python 3.9-3.11, check "Add Python to PATH"
- **Installation fails:** Run as Administrator, check internet connection
- **Slow processing:** Close other apps, consider NVIDIA GPU for 10-20x speedup

## 📧 Support & Contributing

- Issues/questions: [GitHub Issues](https://github.com/BrayanVillatoro/DriveOS/issues)
- Contributions welcome via Pull Requests

---

Made with ❤️ for the racing community | **DriveOS** - Drive faster, smarter, better 🏁
