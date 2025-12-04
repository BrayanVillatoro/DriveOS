import random
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
IMG_DIR = DATA_DIR / 'user_annotations' / 'images'
MASK_DIR = DATA_DIR / 'user_annotations' / 'masks'

def load_pairs(max_samples=50):
    imgs = sorted([p for p in IMG_DIR.glob('*.jpg')])
    if not imgs:
        imgs = sorted([p for p in IMG_DIR.glob('*.png')])
    pairs = []
    for img_path in imgs:
        mask_path = MASK_DIR / img_path.name.replace('.jpg', '.png')
        if not mask_path.exists():
            mask_path = MASK_DIR / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    random.shuffle(pairs)
    return pairs[:max_samples]

def overlay(img, mask):
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    track_mask = (mask > 0).astype(np.uint8)

    # Raw edges from mask
    edges = cv2.Canny(track_mask * 255, 50, 150)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel_edge, iterations=1)

    # Green overlay
    overlay = np.zeros_like(img)
    overlay[:, :, 1] = track_mask * 200
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # White edges on top
    blended[edges > 0] = [255, 255, 255]
    return blended

def main():
    pairs = load_pairs()
    if not pairs:
        print('No image/mask pairs found in', IMG_DIR, MASK_DIR)
        return
    grid = []
    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        vis = overlay(img, mask)
        # Put filename
        cv2.putText(vis, img_path.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        grid.append(vis)

    # Show sequentially
    for vis in grid:
        cv2.imshow('Label Audit', vis)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
