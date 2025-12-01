"""
Interactive tool to create a homography matrix H (image -> world/BEV) and save it
as an .npy file (H.npy). Optionally generates a preview image with a reprojected grid.

Two modes:
1) rectangle: Click 4 points on the image corresponding to a known world rectangle.
              Order: top-left, top-right, bottom-right, bottom-left.
              Provide --width-m and --height-m for the real-world dimensions (meters).
2) points:    Provide a CSV with columns x_img,y_img,x_world,y_world for >=4 correspondences.

Usage examples:
    # Use a still image, click 4 points, world rectangle is 20m x 10m
    python scripts/make_homography.py --image data/user_annotations/images/frame_000000.jpg \
        --mode rectangle --width-m 20 --height-m 10 --output H.npy

    # Use a video (grabs frame at 5.0 seconds)
    python scripts/make_homography.py --video path/to/clip.mp4 --time 5.0 \
        --mode rectangle --width-m 25 --height-m 12 --output H.npy

    # Predefined correspondence CSV
    python scripts/make_homography.py --image path/to/frame.jpg --mode points \
        --points-csv path/to/correspondences.csv --output H.npy

After generating H.npy, you can set HOMOGRAPHY_PATH and run the GUI:
    set HOMOGRAPHY_PATH=c:\\full\\path\\to\\H.npy  (PowerShell: $env:HOMOGRAPHY_PATH = "c:/.../H.npy")
    python -m src.gui
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2


def _load_frame_from_video(video_path: str, time_s: float = 0.0, frame_index: int = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    if frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame at requested position")
    return frame


class ClickCollector:
    def __init__(self, image, max_points=4, window_name='Select 4 points (TL, TR, BR, BL)'):
        self.image = image
        self.points = []
        self.window_name = window_name
        self.max_points = max_points

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.max_points:
            self.points.append((x, y))

    def run(self):
        img_disp = self.image.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.callback)
        while True:
            disp = img_disp.copy()
            for i, (x, y) in enumerate(self.points):
                cv2.circle(disp, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(disp, f"{i}", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord('q')):  # ESC or q cancels
                self.points = []
                break
            if len(self.points) >= self.max_points:
                break
        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.float64)


def compute_homography_rectangle(img_pts: np.ndarray, width_m: float, height_m: float) -> np.ndarray:
    if img_pts.shape != (4, 2):
        raise ValueError("Need exactly 4 image points for rectangle mode")
    # Order must be TL, TR, BR, BL in image
    world_pts = np.array([
        [0.0, 0.0],
        [width_m, 0.0],
        [width_m, height_m],
        [0.0, height_m],
    ], dtype=np.float64)
    H, status = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")
    return H


def compute_homography_from_csv(csv_path: str) -> np.ndarray:
    import csv
    img_pts = []
    world_pts = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        required = {'x_img', 'y_img', 'x_world', 'y_world'}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(f"CSV must have headers: {required}")
        for row in reader:
            img_pts.append([float(row['x_img']), float(row['y_img'])])
            world_pts.append([float(row['x_world']), float(row['y_world'])])
    img_pts = np.array(img_pts, dtype=np.float64)
    world_pts = np.array(world_pts, dtype=np.float64)
    if img_pts.shape[0] < 4:
        raise ValueError("Need at least 4 correspondences in CSV")
    H, status = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("findHomography failed")
    return H


def overlay_grid_preview(frame_bgr: np.ndarray, H: np.ndarray, out_path: str, cell_m: float = 2.0, nx: int = 25, ny: int = 15):
    """Draw a world grid (meters) projected back to the image using H^{-1} and save preview."""
    H_inv = np.linalg.inv(H)
    h, w = frame_bgr.shape[:2]
    # Build grid lines in world coordinates
    max_w = cell_m * nx
    max_h = cell_m * ny
    def to_h(pts):
        pts = np.asarray(pts, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1))
        return np.hstack([pts, ones])
    def from_h(pts_h):
        wv = pts_h[:, 2:3]
        wv[wv == 0] = 1e-12
        return pts_h[:, :2] / wv

    img = frame_bgr.copy()
    # Vertical lines (x constant in world)
    for i in range(nx + 1):
        xw = i * cell_m
        pts_w = np.array([[xw, 0.0], [xw, max_h]], dtype=np.float64)
        pts_i = from_h((H_inv @ to_h(pts_w).T).T)
        pts_i = np.round(pts_i).astype(int)
        cv2.line(img, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 1, cv2.LINE_AA)
    # Horizontal lines (y constant in world)
    for j in range(ny + 1):
        yw = j * cell_m
        pts_w = np.array([[0.0, yw], [max_w, yw]], dtype=np.float64)
        pts_i = from_h((H_inv @ to_h(pts_w).T).T)
        pts_i = np.round(pts_i).astype(int)
        cv2.line(img, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser(description='Create homography H.npy (image -> world).')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', type=str, help='Path to image file')
    src.add_argument('--video', type=str, help='Path to video file')
    ap.add_argument('--time', type=float, default=0.0, help='Timestamp (s) to grab frame from video')
    ap.add_argument('--frame', type=int, default=None, help='Frame index to grab from video')

    ap.add_argument('--mode', type=str, choices=['rectangle', 'points'], default='rectangle')
    ap.add_argument('--width-m', type=float, default=20.0, help='World rectangle width (m)')
    ap.add_argument('--height-m', type=float, default=10.0, help='World rectangle height (m)')
    ap.add_argument('--points-csv', type=str, help='CSV with columns x_img,y_img,x_world,y_world')

    ap.add_argument('--output', type=str, default='H.npy', help='Output path for H.npy')
    ap.add_argument('--preview', type=str, default=None, help='Optional preview image path to save grid overlay')

    args = ap.parse_args()

    # Load frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}")
            sys.exit(1)
    else:
        frame = _load_frame_from_video(args.video, args.time, args.frame)

    if args.mode == 'rectangle':
        print("Click 4 points: top-left, top-right, bottom-right, bottom-left on the planar track surface.")
        clicker = ClickCollector(frame, max_points=4)
        img_pts = clicker.run()
        if img_pts.shape != (4, 2):
            print("Cancelled or not enough points.")
            sys.exit(1)
        H = compute_homography_rectangle(img_pts, args.width_m, args.height_m)
    else:
        if not args.points_csv:
            print("--points-csv is required for mode=points")
            sys.exit(1)
        H = compute_homography_from_csv(args.points_csv)

    np.save(args.output, H)
    print(f"Saved homography to: {args.output}")

    # Optional preview
    if args.preview:
        try:
            overlay_grid_preview(frame, H, args.preview)
            print(f"Saved grid preview to: {args.preview}")
        except Exception as e:
            print(f"Failed to save preview: {e}")


if __name__ == '__main__':
    main()
