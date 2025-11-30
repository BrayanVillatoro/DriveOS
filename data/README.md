# DriveOS Training Data

This directory contains training data for the DriveOS racing line detection model.

## Directory Structure

```
data/
├── sample_training/      # Pre-made training data (included in repo)
├── user_annotations/     # Data created with annotation tool (gitignored)
└── user_data/           # User-generated data (gitignored)
```

## Creating Training Data

You have **3 options** to create training data:

### Option 1: Use Interactive Annotation Tool (Recommended)
The easiest way to create high-quality training data:

1. Open DriveOS GUI
2. Go to the **"✏️ Create Training Data"** tab
3. Select a racing video
4. Use your mouse to draw:
   - Racing line (yellow)
   - Track boundaries (red/blue)
5. Press SPACE to save each frame
6. Annotate 50-100 diverse frames for best results

**Controls:**
- `1/2/3` - Switch between racing line, left boundary, right boundary
- `SPACE` - Save frame
- `N/B` - Next/Previous frame
- `C` - Clear annotations
- `Q` - Quit

### Option 2: Extract from Game Footage
If you have racing game screenshots with visible yellow racing lines:

```bash
# Process a single image
python prepare_sample_data.py path/to/image.jpg --output data/my_training

# Process entire directory
python prepare_sample_data.py path/to/screenshots/ --output data/my_training
```

This automatically detects yellow racing lines and creates masks.

### Option 3: Use Command Line Annotation Tool
For advanced users who prefer command line:

```bash
python -m src.annotate path/to/video.mp4 --output data/my_annotations
```

## Training Data Format

Training data should have this structure:

```
your_data_folder/
├── images/              # RGB images (720x720 or similar)
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── masks/               # Segmentation masks
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
└── annotations/         # (Optional) JSON metadata
    ├── frame_000000.json
    └── ...
```

**Mask Classes:**
- `0` - Background
- `1` - Racing line
- `2` - Track boundaries
- `3` - Track surface

## Using Sample Data

Place your sample training data in `data/sample_training/` to get started quickly.

## Training Tips

### Data Quality
- **Diversity matters**: Include different track types, lighting conditions, and camera angles
- **Minimum samples**: At least 50-100 annotated frames for basic results
- **Optimal samples**: 500-1000+ frames for production-quality model
- **Balance**: Mix of straights, slow corners, fast corners, and complex sections

### Annotation Best Practices
- Draw racing lines through apex of corners
- Be consistent with boundary placement
- Annotate both entry and exit of corners
- Include frames from different parts of track
- Mark both left and right track boundaries when possible

## Training the Model

Once you have training data:

```bash
# Using GUI (easiest)
1. Go to "🎯 Train Model" tab
2. Select your data directory
3. Adjust parameters
4. Click "Start Training"

# Using command line (advanced)
python -m src.train --data-dir data/your_training_folder --epochs 50 --batch-size 4
```

Happy training! 🏁
