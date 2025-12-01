"""Quick audit of training dataset masks and annotations.
Usage:
    python scripts/audit_dataset.py --data-dir data/training
Outputs:
    - Prints class pixel counts and percentages.
    - Lists missing mask files.
    - Writes summary CSV (dataset_audit.csv) to data-dir.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import json
import csv

CLASS_NAMES = {
    0: "track",
    1: "racing_line",
    2: "off_track",
    3: "edge (unused)",
    4: "curb (unused)",
}


def audit(data_dir: Path):
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    anno_dir = data_dir / "annotations"
    if not img_dir.exists():
        raise SystemExit(f"Images dir missing: {img_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Masks dir missing: {mask_dir}")

    image_files = sorted(img_dir.glob("*.jpg"))
    if not image_files:
        raise SystemExit("No images found.")

    class_counts = np.zeros(5, dtype=np.int64)
    missing_masks = []
    racing_line_world_segments = 0

    for img_path in image_files:
        mask_path = mask_dir / img_path.name.replace(".jpg", ".png")
        if not mask_path.exists():
            missing_masks.append(mask_path.name)
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing_masks.append(mask_path.name)
            continue
        # Clip to expected range
        mask = np.clip(mask, 0, 4)
        # Count pixels
        for c in range(5):
            class_counts[c] += int((mask == c).sum())
        # Check annotation JSON for world coords
        anno_path = anno_dir / ("anno_" + img_path.name.replace("frame_", "").replace(".jpg", ".json"))
        # Fallback naming if different pattern
        if not anno_path.exists():
            anno_path = anno_dir / f"anno_{img_path.name.replace('.jpg','')}.json"
        if anno_path.exists():
            try:
                with open(anno_path, 'r') as f:
                    data = json.load(f)
                if 'racing_line_world' in data and data['racing_line_world']:
                    racing_line_world_segments += len(data['racing_line_world'])
            except Exception:
                pass

    total_pixels = class_counts.sum()
    print("==== DATASET AUDIT ====\n")
    print(f"Images: {len(image_files)}")
    print(f"Missing masks: {len(missing_masks)}")
    if missing_masks:
        print("First 10 missing:", missing_masks[:10])
    print()
    for c in range(5):
        pct = (class_counts[c] / total_pixels * 100.0) if total_pixels > 0 else 0.0
        print(f"Class {c} ({CLASS_NAMES[c]}): {class_counts[c]:,} px ({pct:.2f}%)")
    print(f"Frames with world racing line segments: {racing_line_world_segments}")

    # Write CSV summary
    csv_path = data_dir / "dataset_audit.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["class", "name", "pixels", "percent"])
        for c in range(5):
            pct = (class_counts[c] / total_pixels * 100.0) if total_pixels > 0 else 0.0
            w.writerow([c, CLASS_NAMES[c], class_counts[c], f"{pct:.4f}"])
    print(f"\nWrote summary: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    args = ap.parse_args()
    audit(Path(args.data_dir))
