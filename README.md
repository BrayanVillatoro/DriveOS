# DriveOS

DriveOS is an intelligent racing-line analysis system that uses machine learning to identify the fastest, most efficient path around any track. It processes onboard video and telemetry to deliver real-time insights that help drivers improve lap times and racecraft.

## Features

- 🎥 **Video Analysis**: Process onboard racing video to detect track boundaries and racing lines
- 📊 **Telemetry Processing**: Analyze speed, throttle, brake, steering, and other telemetry data
- 🤖 **ML-Powered Predictions**: Deep learning models for optimal racing line prediction
- ⚡ **Real-Time Processing**: Low-latency inference for live analysis
- 📈 **Performance Insights**: Actionable feedback for driver improvement
- 🔄 **Lap Comparison**: Compare telemetry between different laps
- 🌐 **REST API**: Easy integration with other applications
- 🐳 **Docker Support**: Simple deployment with containerization

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

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Disclaimer

**IMPORTANT: This is research software intended for educational and research purposes only.**

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors of DriveOS are not liable for any injuries, damages, losses, or accidents that may occur from the use or misuse of this software. 

Racing and high-performance driving are inherently dangerous activities. This software is NOT a substitute for:
- Professional driving instruction
- Proper safety equipment and procedures
- Track safety regulations and guidelines
- Sound judgment and personal responsibility

Users assume all risks associated with racing activities. Always prioritize safety and follow all applicable laws, regulations, and track rules. The insights and recommendations provided by this software should be reviewed by qualified professionals before implementation.

By using this software, you acknowledge and accept these terms and agree to hold harmless the authors, contributors, and associated parties from any and all claims, damages, or liabilities.

## License

MIT License - see LICENSE file for details

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
