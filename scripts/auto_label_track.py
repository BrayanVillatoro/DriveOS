"""
Automatic Track Edge Detection for Initial Training Data

This script uses computer vision to automatically detect:
- Track edges (green) using edge detection
- Track surface vs grass (using color/texture detection)
- Approximate racing line (center of track)

While not perfect, this provides a starting point that you can refine manually.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class AutoTrackLabeler:
    """Automatically label track features using computer vision"""
    
    def __init__(self):
        pass
    
    def detect_track_edges(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect track edges using color segmentation and edge detection
        
        Returns:
            Binary mask of track edges
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect track surface (usually gray/dark)
        # Track is typically low saturation (not very colorful)
        low_sat = cv2.inRange(hsv, (0, 0, 30), (180, 80, 200))
        
        # Apply morphology to clean up
        kernel = np.ones((5, 5), np.uint8)
        track_mask = cv2.morphologyEx(low_sat, cv2.MORPH_CLOSE, kernel)
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel)
        
        # Detect edges
        edges = cv2.Canny(track_mask, 50, 150)
        
        # Dilate edges to make them more visible
        edge_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, edge_kernel, iterations=2)
        
        return edges, track_mask
    
    def detect_racing_line(self, track_mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Approximate racing line as path through track center
        
        Returns:
            Points defining the racing line
        """
        h, w = track_mask.shape
        
        # Find track center at different vertical positions
        racing_line_points = []
        
        # Sample horizontal slices through the image
        for y in range(h // 4, 3 * h // 4, 10):
            # Find track pixels at this height
            row = track_mask[y, :]
            track_pixels = np.where(row > 0)[0]
            
            if len(track_pixels) > 10:
                # Use weighted center (bias toward inside of turns)
                center_x = int(np.mean(track_pixels))
                racing_line_points.append((center_x, y))
        
        return np.array(racing_line_points) if racing_line_points else None
    
    def create_segmentation_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create full segmentation mask with all track features
        
        Mask classes:
        0 = Track surface
        1 = Optimal racing line
        2 = Off-track (grass/gravel)
        3 = Track edge
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Detect track features
        edges, track_mask = self.detect_track_edges(frame)
        
        # Set track surface (class 0) - already default
        
        # Set off-track areas (class 2) - everything not track
        mask[track_mask == 0] = 2
        
        # Set track edges (class 3)
        mask[edges > 0] = 3
        
        # Set racing line (class 1)
        racing_line = self.detect_racing_line(track_mask, edges)
        if racing_line is not None and len(racing_line) > 0:
            # Draw racing line on mask
            for i in range(len(racing_line) - 1):
                pt1 = tuple(racing_line[i])
                pt2 = tuple(racing_line[i + 1])
                cv2.line(mask, pt1, pt2, 1, thickness=8)
        
        return mask
    
    def visualize_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create visualization of the segmentation mask"""
        overlay = frame.copy()
        
        # Color code the mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 1] = [255, 0, 255]  # Purple - racing line
        colored_mask[mask == 2] = [0, 0, 255]    # Red - off-track
        colored_mask[mask == 3] = [0, 255, 0]    # Green - track edges
        
        # Blend
        result = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        return result
    
    def process_video(self, video_path: str, output_dir: str, 
                     frame_interval: int = 30, preview: bool = True):
        """
        Process video and generate training data automatically
        
        Args:
            video_path: Input video path
            output_dir: Output directory for training data
            frame_interval: Extract every N frames
            preview: Show preview of detections
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'masks').mkdir(exist_ok=True)
        (output_path / 'previews').mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        saved_count = 0
        
        print("Processing video... Press 'q' to quit, 's' to skip, any other key to continue")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Generate mask
                mask = self.create_segmentation_mask(frame)
                
                # Visualize
                viz = self.visualize_mask(frame, mask)
                
                if preview:
                    cv2.imshow('Auto-labeled Track', viz)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        frame_count += 1
                        continue
                
                # Save
                frame_id = saved_count
                cv2.imwrite(str(output_path / 'images' / f'frame_{frame_id:06d}.jpg'), frame)
                cv2.imwrite(str(output_path / 'masks' / f'mask_{frame_id:06d}.png'), mask)
                cv2.imwrite(str(output_path / 'previews' / f'preview_{frame_id:06d}.jpg'), viz)
                
                saved_count += 1
                print(f"Saved frame {saved_count} (source frame {frame_count})")
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n=== Complete ===")
        print(f"Generated {saved_count} training samples")
        print(f"Saved to: {output_dir}")
        print(f"\nRecommendation:")
        print(f"1. Review the previews in {output_dir}/previews")
        print(f"2. Refine labels manually using prepare_training_data.py if needed")
        print(f"3. Train the model with: python -m src.main train --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Automatically generate track edge labels'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to racing video')
    parser.add_argument('--output-dir', type=str, default='data/auto_labels',
                       help='Output directory')
    parser.add_argument('--interval', type=int, default=30,
                       help='Extract every N frames')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    
    args = parser.parse_args()
    
    labeler = AutoTrackLabeler()
    labeler.process_video(
        args.video, 
        args.output_dir, 
        args.interval,
        preview=not args.no_preview
    )


if __name__ == '__main__':
    main()
