# Racing Line Annotation Guide for DriveOS

## Overview
This guide helps you create high-quality training data for the racing line detection model. Proper annotations are crucial for the AI to learn correct racing behavior.

## 🎯 Core Principles

### 1. **Racing Line Theory**

The optimal racing line follows these principles:

#### Approaching Corners
- **LEFT TURN**: Position car on **RIGHT side** of track (outside)
- **RIGHT TURN**: Position car on **LEFT side** of track (outside)
- **Goal**: Maximize corner radius for higher speed

#### Corner Apex
- Hit the **LATE APEX** (past the geometric center of the corner)
- Apex point is the closest point to inside edge of corner
- Late apex allows earlier throttle application

#### Corner Exit
- Use **FULL TRACK WIDTH** on exit
- Let car drift to outside edge naturally
- Maximize exit speed for following straight

#### Straights
- Use **CENTER** of track unless setting up for next corner
- Begin positioning early (100-200m before corner)

### 2. **Annotation Classes**

#### Class 0: Track Surface (Green)
- All drivable racing surface
- Include painted runoff areas
- Exclude grass, gravel, walls

#### Class 1: Racing Line (Purple)
- The OPTIMAL path, not just any path
- Follow racing theory (see above)
- Should be smooth and continuous
- Width: ~8-15 pixels in 320x320 images

#### Class 2: Off-Track (Red)
- Grass, gravel, walls
- Non-drivable areas
- Paint broadly to help model understand boundaries

#### Class 3: Track Edges (Green lines)
- Physical boundaries of track
- Curbs that mark edge
- Use for clear definition

#### Class 4: Curbs (Blue)
- Rumble strips / kerbs
- Often rideable but not optimal

## 📝 Annotation Workflow

### Frame Selection
1. **Variety is Key**: Annotate diverse scenarios
   - Different corners (left, right, hairpin, sweeper)
   - Different track sections (straights, esses, chicanes)
   - Different lighting conditions
   - Different track positions

2. **Minimum Dataset Size**:
   - Start with at least **50 frames**
   - Better: **100-150 frames** for robust model
   - Best: **200+ frames** for production quality

3. **Frame Spacing**:
   - Extract every 30 frames (1 per second at 30fps)
   - Avoid too similar frames
   - Capture full corner sequences

### Drawing the Racing Line

1. **Look Ahead**: Understand where the track is going
2. **Plan the Line**: Visualize the optimal path before drawing
3. **Start Wide**: Begin from outside before corner
4. **Late Apex**: Hit the inside at ~60-70% through corner
5. **Exit Wide**: Finish on outside edge

### Quality Checks

Before saving each frame, verify:

- [ ] Racing line follows proper racing theory
- [ ] Racing line is ON the track surface (class 0)
- [ ] Racing line is smooth and continuous
- [ ] Track boundaries are clearly marked
- [ ] Off-track areas are marked where visible
- [ ] No gaps or holes in track surface

## 🎨 Visual Examples

### ✅ Good Annotation Example

```
Track view from behind car, approaching left turn:

                Off-track (Red)
                     ||
        Track Edge ──╔═════════════════════════════╗── Track Edge
                     ║░░░░░░░░░░░░░░░░░░░░░░░░░░░░║
    Car Position ────●═══════════════════════════○── Racing Line
                     ║░░░░░░░░░░░░░░░░░░░░░░░░░░░║    (starts right)
                     ║░░░░░░░░░░░░░░░░░░░░░○░░░░░║
                     ║░░░░░░░░░░░░░░░░░░░○░░░░░░░║    (cuts to apex)
                     ║░░░░░░░░░░░░░░░░░○░░░░░░░░░║
                     ║░░░░░░░░░░░░░░░○░░░░░░░░░░░║
                     ╚═══════════════════════════╝
                          Track Surface (Green)
```

### ❌ Bad Annotation Example

```
Track view from behind car, approaching left turn:

                Off-track (Red)
                     ||
        Track Edge ──╔═════════════════════════════╗
                     ║░░░░░░░░░░░░░░░░░░░░░░░░░░░░║
    Car Position ────●░░░░░░░░░░░░░░░░░░░░░░░░░░░║
                     ║○░░░░░░░░░░░░░░░░░░░░░░░░░░║ ⚠️ Line starts
                     ║░○░░░░░░░░░░░░░░░░░░░░░░░░░║    on LEFT (wrong!)
                     ║░░○░░░░░░░░░░░░░░░░░░░░░░░░║
                     ║░░░○░░░░░░░░░░░░░░░░░░░░░░░║ ⚠️ Hugging inside
                     ║░░░░○═══════════════════════║    (slow!)
                     ╚═══════════════════════════╝
```

## 🔧 Common Mistakes to Avoid

### 1. **Centerline Bias**
❌ Drawing line down geometric center of track
✅ Position line based on upcoming corners

### 2. **Early Apex**
❌ Hitting apex too early in corner
✅ Late apex allows better exit speed

### 3. **Tight Corners**
❌ Staying in middle of track through corners
✅ Use full track width (outside-inside-outside)

### 4. **Disconnected Track**
❌ Gaps in track surface annotation
✅ Ensure continuous track surface

### 5. **Line Off Track**
❌ Racing line going over curbs or off-track
✅ Keep racing line on drivable surface

## 📊 Validation Tips

Use the built-in validation when saving:

```
⚠️  No racing line annotated         → Add racing line
⚠️  Racing line appears to be off track! → Redraw on track
⚠️  Track surface is very small       → Annotate more track
```

Fix warnings before proceeding to next frame.

## 🚀 Advanced Tips

### 1. **Use Reference Footage**
- Watch real racing onboards (F1, GT3, etc.)
- Study racing simulators (iRacing, ACC)
- Learn from track guides

### 2. **Corner Classification**

| Corner Type | Entry Position | Apex | Exit |
|------------|---------------|------|------|
| Hairpin | Very wide | Late | Full width |
| Sweeper | Slight wide | Mid | Gentle drift |
| Chicane | Setup for 2nd | Early 1st, late 2nd | Compromise |
| Esses | Flow through | Rhythm | Connect exits |

### 3. **Context Awareness**
- Faster corner → Earlier positioning
- Slower corner → Later positioning
- Following straight → Prioritize exit
- Following corner → Prioritize entry

## 📈 Training Recommendations

### Dataset Balance
- 40% straights and approach sections
- 40% corners (various types)
- 20% complex sections (chicanes, esses)

### Augmentation
After manual annotation, use augmentation script:
```bash
python scripts/augment_training_data.py \
    --input-dir data/user_annotations \
    --output-dir data/augmented \
    --multiplier 5
```

This will 5x your dataset with variations.

### Iterative Improvement
1. Annotate 50 frames
2. Train model (25 epochs)
3. Test on validation video
4. Identify failures
5. Add 25 more frames targeting failures
6. Retrain
7. Repeat until satisfied

## 🎓 Resources

### Racing Theory
- "Going Faster! Mastering the Art of Race Driving" by Skip Barber
- "Speed Secrets" series by Ross Bentley
- YouTube: "Driver61", "Scott Mansell"

### Technical Setup
```bash
# Annotate frames
python scripts/prepare_training_data.py \
    --video "videos/racing_footage.mp4" \
    --output-dir "data/user_annotations" \
    --interval 30

# Augment dataset
python scripts/augment_training_data.py \
    --input-dir data/user_annotations \
    --output-dir data/training_augmented \
    --multiplier 5

# Train model
python -m src.train \
    --data-dir data/training_augmented \
    --epochs 50 \
    --batch-size 8
```

## 💡 Summary Checklist

Before starting annotation session:
- [ ] Read this guide thoroughly
- [ ] Understand racing line principles
- [ ] Watch reference racing footage
- [ ] Set up annotation tool with proper video

During annotation:
- [ ] Think like a racing driver
- [ ] Plan line before drawing
- [ ] Verify each frame before saving
- [ ] Take breaks to maintain quality

After annotation:
- [ ] Validate dataset (check for errors)
- [ ] Augment data 5x
- [ ] Train model on good hardware
- [ ] Test and iterate

---

**Remember**: Quality > Quantity. 50 well-annotated frames with proper racing lines are better than 200 frames with centerline bias!
