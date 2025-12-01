"""
Training Data Preparation Tool for DriveOS Racing Line Detection

This tool helps you prepare labeled training data for the racing line detection model.
You'll need to manually label a few video frames to show the model what to learn.

Steps:
1. Extract frames from your racing videos
2. Label each frame with:
   - Track edges (green)
   - Optimal racing line (purple) 
   - Track curbs (blue)
   - Off-track areas (red)
3. Save labeled masks
4. Train the model

Usage:
    python prepare_training_data.py --video "path/to/video.mp4" --output-dir "data/training"
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional

def _to_homogeneous(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones])


def _from_homogeneous(pts_h: np.ndarray) -> np.ndarray:
    pts_h = np.asarray(pts_h, dtype=np.float64)
    w = pts_h[:, 2:3]
    w[w == 0] = 1e-12
    return pts_h[:, :2] / w


def _draw_world_grid_overlay(image: np.ndarray, H: np.ndarray, cell_m: float = 2.0, nx: int = 25, ny: int = 15) -> np.ndarray:
    """Return a copy of image with a projected world grid overlay using H (image->world)."""
    if H is None:
        return image
    H_inv = np.linalg.inv(H)
    img = image.copy()
    # Build grid lines in world coordinates
    max_w = cell_m * nx
    max_h = cell_m * ny
    # verticals
    for i in range(nx + 1):
        xw = i * cell_m
        pts_w = np.array([[xw, 0.0], [xw, max_h]], dtype=np.float64)
        pts_i = _from_homogeneous((H_inv @ _to_homogeneous(pts_w).T).T)
        pts_i = np.round(pts_i).astype(int)
        cv2.line(img, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 1, cv2.LINE_AA)
    # horizontals
    for j in range(ny + 1):
        yw = j * cell_m
        pts_w = np.array([[0.0, yw], [max_w, yw]], dtype=np.float64)
        pts_i = _from_homogeneous((H_inv @ _to_homogeneous(pts_w).T).T)
        pts_i = np.round(pts_i).astype(int)
        cv2.line(img, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 1, cv2.LINE_AA)
    return img


class TrainingDataLabeler:
    """Interactive tool for labeling racing video frames"""
    
    def __init__(self, output_dir: str, homography_path: Optional[str] = None, grid_cell_m: float = 2.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.homography_path = homography_path
        self.H = None
        if homography_path and os.path.isfile(homography_path):
            try:
                self.H = np.load(homography_path)
            except Exception:
                self.H = None
        self.grid_cell_m = grid_cell_m
        self.show_grid = True if self.H is not None else False
        
        # Annotation state
        self.current_tool = 'racing_line'  # racing_line, track_edge, curb, off_track
        self.drawing = False
        self.points = []
        self.annotations = {
            'racing_line': [],      # Purple line - optimal path
            'track_edges': [],      # Green lines - track boundaries
            'curbs': [],            # Blue lines - curb edges
            'off_track': []         # Red areas - grass/gravel
        }
        
        # Colors for visualization (BGR format)
        self.colors = {
            'racing_line': (255, 0, 255),   # Magenta/Purple
            'track_edge': (0, 255, 0),      # Green
            'curb': (255, 0, 0),            # Blue
            'off_track': (0, 0, 255)        # Red
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing annotations"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.points) > 1:
                # Save the annotation
                self.annotations[self.current_tool].append(self.points.copy())
                self.points = []
    
    def create_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create segmentation mask from annotations
        
        Mask classes:
        0 = Track surface (default)
        1 = Optimal racing line
        2 = Off-track (grass/gravel)
        3 = Track edge/boundary
        4 = Curb
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw off-track areas (fill polygons)
        for polygon in self.annotations['off_track']:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 2)
        
        # Draw curbs (thick lines)
        for line in self.annotations['curbs']:
            pts = np.array(line, dtype=np.int32)
            cv2.polylines(mask, [pts], False, 4, thickness=15)
        
        # Draw track edges (medium lines)
        for line in self.annotations['track_edges']:
            pts = np.array(line, dtype=np.int32)
            cv2.polylines(mask, [pts], False, 3, thickness=10)
        
        # Draw racing line (thin line)
        for line in self.annotations['racing_line']:
            pts = np.array(line, dtype=np.int32)
            cv2.polylines(mask, [pts], False, 1, thickness=8)
        
        return mask
    
    def label_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """
        Interactive labeling interface for a single frame
        
        Returns:
            True if frame was saved, False if skipped
        """
        display = frame.copy()
        cv2.namedWindow('Label Frame')
        cv2.setMouseCallback('Label Frame', self.mouse_callback)
        
        print("\n=== Frame Labeling Tool ===")
        print("Keys:")
        print("  1 - Draw Racing Line (purple)")
        print("  2 - Draw Track Edges (green)")
        print("  3 - Draw Curbs (blue)")
        print("  4 - Mark Off-Track Areas (red)")
        print("  u - Undo last annotation")
        print("  g - Toggle world grid overlay (if homography loaded)")
        print("  c - Clear all annotations")
        print("  s - Save and next frame")
        print("  n - Skip this frame")
        print("  q - Quit")
        
        while True:
            # Draw current annotations
            display = frame.copy()
            if self.show_grid and self.H is not None:
                try:
                    display = _draw_world_grid_overlay(display, self.H, self.grid_cell_m)
                except Exception:
                    pass
            
            # Draw saved annotations
            for tool, color in self.colors.items():
                for line in self.annotations[tool]:
                    pts = np.array(line, dtype=np.int32)
                    cv2.polylines(display, [pts], False, color, 3)
            
            # Draw current line being drawn
            if len(self.points) > 1:
                color = self.colors[self.current_tool]
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(display, [pts], False, color, 3)
            
            # Show current tool
            tool_text = f"Tool: {self.current_tool.replace('_', ' ').title()}"
            cv2.putText(display, tool_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Label Frame', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                self.current_tool = 'racing_line'
            elif key == ord('2'):
                self.current_tool = 'track_edges'
            elif key == ord('3'):
                self.current_tool = 'curbs'
            elif key == ord('4'):
                self.current_tool = 'off_track'
            elif key == ord('u'):
                # Undo last annotation
                if self.annotations[self.current_tool]:
                    self.annotations[self.current_tool].pop()
            elif key == ord('c'):
                # Clear all annotations
                self.annotations = {k: [] for k in self.annotations}
            elif key == ord('g'):
                self.show_grid = not self.show_grid
            elif key == ord('s'):
                # Save frame and mask
                self.save_labeled_frame(frame, frame_id)
                cv2.destroyWindow('Label Frame')
                return True
            elif key == ord('n'):
                # Skip frame
                cv2.destroyWindow('Label Frame')
                return False
            elif key == ord('q'):
                cv2.destroyWindow('Label Frame')
                return None
    
    def save_labeled_frame(self, frame: np.ndarray, frame_id: int):
        """Save labeled frame and corresponding mask"""
        # Save original frame
        frame_path = self.output_dir / 'images' / f'frame_{frame_id:06d}.jpg'
        frame_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(frame_path), frame)
        
        # Create and save mask
        mask = self.create_mask(frame.shape[:2])
        mask_path = self.output_dir / 'masks' / f'mask_{frame_id:06d}.png'
        mask_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        
        # Save annotations as JSON for reference
        # If homography available, also save racing_line in world coordinates
        annotations_world = None
        if self.H is not None and self.annotations['racing_line']:
            try:
                annotations_world = {
                    'racing_line_world': [
                        _from_homogeneous((self.H @ _to_homogeneous(np.array(line, dtype=np.float64)).T).T).tolist()
                        for line in self.annotations['racing_line'] if len(line) >= 2
                    ]
                }
            except Exception:
                annotations_world = None

        json_path = self.output_dir / 'annotations' / f'anno_{frame_id:06d}.json'
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, 'w') as f:
            payload = {**self.annotations}
            if annotations_world:
                payload.update(annotations_world)
            if self.homography_path:
                payload['homography_path'] = str(Path(self.homography_path).name)
            json.dump(payload, f)
        
        print(f"✓ Saved frame {frame_id}")
        
        # Reset annotations for next frame
        self.annotations = {k: [] for k in self.annotations}


def extract_and_label_frames(video_path: str, output_dir: str, 
                             frame_interval: int = 30,
                             homography_path: Optional[str] = None,
                             grid_cell_m: float = 2.0):
    """
    Extract frames from video and label them interactively
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save training data
        frame_interval: Extract every N frames (default: 30 = ~1 frame per second)
    """
    labeler = TrainingDataLabeler(output_dir, homography_path=homography_path, grid_cell_m=grid_cell_m)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n=== Video Info ===")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Will extract every {frame_interval} frames")
    if homography_path and os.path.isfile(homography_path):
        print(f"Using homography: {homography_path} (grid {grid_cell_m} m)")
        # Save a copy in dataset root for convenience
        try:
            H = np.load(homography_path)
            np.save(str(Path(output_dir) / 'homography.npy'), H)
        except Exception:
            pass
    print(f"Estimated frames to label: {total_frames // frame_interval}")
    
    frame_count = 0
    labeled_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            print(f"\nFrame {frame_count}/{total_frames}")
            result = labeler.label_frame(frame, frame_count)
            
            if result is None:  # Quit
                break
            elif result:  # Saved
                labeled_count += 1
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Complete ===")
    print(f"Labeled {labeled_count} frames")
    print(f"Saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review your labeled data in {output_dir}/images and {output_dir}/masks")
    print(f"2. Label at least 50-100 diverse frames for good results")
    print(f"3. Run training: python -m src.main train --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for racing line detection'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to racing video')
    parser.add_argument('--output-dir', type=str, default='data/training',
                       help='Output directory for training data')
    parser.add_argument('--interval', type=int, default=30,
                       help='Extract every N frames (default: 30)')
    parser.add_argument('--homography', type=str, default=None,
                       help='Optional path to H.npy (image->world) for grid overlay and world annotations')
    parser.add_argument('--grid-cell-m', type=float, default=2.0,
                       help='Grid cell size in meters when H is provided (default: 2m)')
    
    args = parser.parse_args()
    
    extract_and_label_frames(
        args.video,
        args.output_dir,
        args.interval,
        homography_path=args.homography,
        grid_cell_m=args.grid_cell_m,
    )


if __name__ == '__main__':
    main()
