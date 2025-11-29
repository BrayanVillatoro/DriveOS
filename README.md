# 🏁 DriveOS - AI Racing Line Analyzer

**AI-powered racing line analysis to find the fastest, most efficient path around any track**

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

## 🚀 Quick Start (Recommended for Most Users)

### Easy Installation

1. **Download** this repository (click "Code" → "Download ZIP")
2. **Extract** the ZIP file to any location
3. **Double-click** `INSTALL.bat` to start the automatic installation
4. Wait for the installation to complete (~2-5 minutes)
5. **Launch** DriveOS from your Desktop shortcut!

That's it! The installer will automatically:
- ✅ Check your Python version
- ✅ Create a virtual environment
- ✅ Install PyTorch and all dependencies
- ✅ Create a Desktop shortcut
- ✅ Set everything up for you

> **Note:** If you prefer a GUI installer with more options, run `python installer.py` after installing dependencies

### System Requirements

- **Operating System:** Windows 10 or newer
- **Python:** 3.9, 3.10, or 3.11 ([Download here](https://www.python.org/downloads/))
  - ⚠️ Make sure to check "Add Python to PATH" during Python installation
- **Disk Space:** ~2 GB for installation
- **RAM:** 8 GB minimum (16 GB recommended)
- **CPU:** Multi-core processor (16+ threads recommended for CPU processing)

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

## 🛠️ Advanced Installation (For Developers)

If you want to modify the code or contribute to the project:

```bash
# Clone the repository
git clone https://github.com/BrayanVillatoro/DriveOS.git
cd DriveOS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows

# Install PyTorch (CPU version)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# (Optional) Install screen capture support
pip install mss

# Launch the GUI
python launch_gui.py
```

## 🧠 Technical Details

### Architecture

- **Vision Model:** DeepLabV3 with ResNet50 backbone
- **Telemetry Model:** LSTM for temporal sequence analysis
- **Fusion Model:** Combines vision and telemetry predictions
- **Training:** Supervised learning with auto-generated labels
- **Inference:** Optimized for CPU with 16-thread processing

### Performance

- **CPU Processing:** ~30 FPS on 16-thread CPU
- **Resolution:** 320x320 optimized for speed
- **GPU Support:** Currently optimized for CPU (GPU support for newer cards coming in PyTorch 2.9+)

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Architecture

DriveOS combines computer vision and time-series analysis:

- **Vision Models**: DeepLabV3 + ResNet50 for track segmentation
- **Telemetry Models**: LSTM networks for sequential data prediction
- **Fusion**: Combined vision-telemetry model for optimal line detection

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, for faster inference)
- FFmpeg (for video processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BrayanVillatoro/DriveOS.git
cd DriveOS
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Command Line Interface

**Analyze a racing video:**
```bash
python -m src.main analyze path/to/video.mp4 --telemetry path/to/telemetry.csv --output analyzed_video.mp4
```

**Analyze telemetry only:**
```bash
python -m src.main telemetry path/to/telemetry.csv --output report.html
```

**Compare two laps:**
```bash
python -m src.main compare lap1.csv lap2.csv --output comparison.html
```

**Start API server:**
```bash
python -m src.main serve --host 0.0.0.0 --port 8000
```

### API Server

Start the API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /analyze/video` - Analyze racing video with telemetry
- `POST /telemetry/analyze` - Analyze telemetry data
- `GET /insights/compare` - Compare two laps
- `WS /ws/realtime` - WebSocket for real-time analysis

Visit `http://localhost:8000/docs` for interactive API documentation.

### Docker Deployment

Build and run with Docker Compose:
```bash
docker-compose up -d
```

Services:
- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Nginx: `http://localhost:80`

## Project Structure

```
DriveOS/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── video_processor.py     # Video processing utilities
│   ├── telemetry_processor.py # Telemetry analysis
│   ├── models.py              # ML model architectures
│   ├── inference.py           # Inference engine
│   ├── api.py                 # FastAPI application
│   ├── visualization.py       # Visualization utilities
│   └── main.py                # CLI application
├── models/                    # Trained model weights
├── data/                      # Sample data
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Telemetry Data Format

CSV format with the following columns:
- `timestamp` - Time in seconds
- `speed` - Speed in km/h
- `throttle` - Throttle position (0-100%)
- `brake` - Brake pressure (0-100%)
- `steering` - Steering angle (-100 to 100)
- `gear` - Current gear
- `rpm` - Engine RPM
- `latitude` - GPS latitude (optional)
- `longitude` - GPS longitude (optional)

Example:
```csv
timestamp,speed,throttle,brake,steering,gear,rpm
0.0,120.5,85.0,0.0,15.2,4,7500
0.01,122.3,90.0,0.0,12.1,4,7800
```

## Model Training

To train custom models:

1. Prepare your dataset (video + telemetry pairs)
2. Create a training notebook in `notebooks/`
3. Use the model classes from `src/models.py`
4. Save trained weights to `models/`

Example training script coming soon!

## Performance

- **Video Processing**: ~30 FPS on GPU, ~5 FPS on CPU
- **Inference Time**: 15-30ms per frame (GPU), 100-200ms (CPU)
- **Memory Usage**: ~2GB GPU VRAM, ~4GB RAM

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

## Acknowledgments

- PyTorch and torchvision for ML frameworks
- OpenCV for video processing
- FastAPI for API framework
- Plotly for visualizations

## Support

For issues and questions:
- GitHub Issues: https://github.com/BrayanVillatoro/DriveOS/issues
- Documentation: Coming soon!

## Hardware Recommendations

### Recording Equipment

**Video Recording:**
- **GoPro Hero 11/12 Black** - Excellent stabilization, 4K@60fps, wide FOV perfect for cockpit mounting
- **DJI Osmo Action 4** - Great low-light performance, RockSteady stabilization
- **Insta360 X3** - 360° capture lets you choose angle in post-production
- **Budget option**: GoPro Hero 9/10 or phone with good stabilization

**Mounting**: Windscreen suction mount or roll cage mount for stable cockpit view showing track ahead

**Telemetry Recording:**
- **RaceBox Mini S** (~$200) - 25Hz GPS, accelerometer, great for amateur racing
- **AiM Solo 2** (~$400) - 10Hz GPS, integrates with OBD-II, LCD display
- **VBOX Sport** (~$500) - 20Hz GPS, professional-grade accuracy
- **RaceChrono + phone** (Free app) - Uses phone GPS/sensors, exports CSV
- **Budget option**: Phone with RaceChrono Pro app ($10-30)

**OBD-II Data Loggers:**
- **OBDLink MX+** - Bluetooth adapter for reading ECU data (throttle, RPM, etc.)
- **RaceCapture** - Advanced logger with GPS + OBD-II integration

### Custom Raspberry Pi Solution

Build your own data acquisition system for ~$225:

**Hardware Components:**
- **Raspberry Pi 5 (8GB)** (~$80) - Best performance for ML inference
  - Alternative: Pi 4 (8GB) (~$75) if budget constrained
- **Camera Module 3** (~$25) - 12MP, great low-light performance, autofocus
  - Alternative: HQ Camera with wide-angle lens for better field of view
- **GPS Module**: u-blox NEO-M9N (~$40) - 25Hz update rate for accurate positioning
- **IMU/Accelerometer**: MPU-9250 or BMI088 (~$15) - Measure G-forces and orientation
- **OBD-II HAT** (~$30) - Read ECU data directly from car
- **Power Supply**: 12V to 5V converter (~$15) - Reliable car power conversion
- **Storage**: 128GB+ microSD card (~$20) - Store video and telemetry data

**Custom PCB HAT Features:**
- Integrated GPS module
- IMU/accelerometer
- OBD-II interface
- Power management circuit
- Camera connector
- Optional display output

**Advantages:**
- **Cost-effective**: Significantly cheaper than commercial solutions
- **Customizable**: Full control over sampling rates and data formats
- **Integrated**: Run DriveOS inference directly on device
- **Real-time feedback**: Display insights while driving
- **Open source**: No vendor lock-in, expandable with additional sensors
- **Flexible processing**: Record locally, process post-session or real-time with lightweight models

**Setup Considerations:**
- Cooling solution required (Pi 5 runs hot in enclosed environments)
- Clean 5V 3A+ power supply essential
- Secure mounting with vibration dampening
- Consider lightweight model for real-time inference, full analysis post-session on GPU

### Recording Tips

1. **Sync timing**: Start video and telemetry recording simultaneously for easier alignment
2. **CSV export**: Ensure telemetry device exports to DriveOS-compatible CSV format
3. **Frame rate**: Use 30-60fps for smooth analysis (higher = more processing time)
4. **Storage**: 128GB+ recommended for extended track sessions

**Budget Setup**: GoPro Hero 9/10 + RaceBox Mini S (~$400 total)  
**Professional Setup**: GoPro Hero 12 + AiM Solo 2 (~$900 total)  
**DIY Setup**: Custom Raspberry Pi system (~$225 total)

## Roadmap

- [ ] Pre-trained models for common tracks
- [ ] Mobile app integration
- [ ] Live streaming support
- [ ] Multi-car comparison
- [ ] Track database
- [ ] Cloud deployment guides
- [ ] Web dashboard UI
- [ ] Raspberry Pi optimized models
- [ ] Custom hardware PCB design files

---

**DriveOS** - Drive faster, smarter, better. 🏁
