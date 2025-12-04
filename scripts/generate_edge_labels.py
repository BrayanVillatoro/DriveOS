"""
Generate edge polyline annotations from segmentation masks
Following openpilot's data format
"""
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional, List
from tqdm import tqdm

from src.edge_constants import EdgeConstants


def extract_edge_polylines_from_mask(mask: np.ndarray, 
                                      num_points: int = EdgeConstants.IDX_N,
                                      img_height: int = 320,
                                      img_width: int = 320) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract left/right edge polylines from binary segmentation mask
    
    Args:
        mask: Binary mask [H, W] where >0 is drivable
        num_points: Number of points to sample along each edge
        img_height: Image height for normalization
        img_width: Image width for normalization
    
    Returns:
        Tuple of (left_edge, right_edge) as [N, 2] arrays (y_offset, height)
        or (None, None) if edges cannot be extracted
    """
    # Ensure binary
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None, None
    
    # Use largest contour as track boundary
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < num_points:
        return None, None
    
    # Reshape contour
    contour = contour.reshape(-1, 2)  # [N, 2] -> (x, y)
    
    # Split into left/right by x-coordinate relative to center
    center_x = img_width // 2
    left_mask = contour[:, 0] < center_x
    right_mask = contour[:, 0] >= center_x
    
    left_pts = contour[left_mask]
    right_pts = contour[right_mask]
    
    if len(left_pts) < 10 or len(right_pts) < 10:
        return None, None
    
    # Sort by y (top to bottom)
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    
    # Sample points evenly with quadratic spacing
    # (denser near bottom/car, sparser far ahead)
    def sample_edge(pts, n_samples):
        if len(pts) < n_samples:
            # Interpolate if not enough points
            indices = np.linspace(0, len(pts) - 1, n_samples)
            indices_int = indices.astype(int)
            return pts[indices_int]
        else:
            # Use quadratic spacing for sampling
            max_idx = len(pts) - 1
            indices = [int(max_idx * (1 - (i / (n_samples - 1))**2)) for i in range(n_samples)]
            indices = np.clip(indices, 0, max_idx)
            return pts[indices]
    
    left_edge_pts = sample_edge(left_pts, num_points)
    right_edge_pts = sample_edge(right_pts, num_points)
    
    # Convert to normalized coordinates
    # x -> lateral offset from center (in pixels, then normalize to meters)
    # y -> height (in pixels from bottom, normalized)
    
    def normalize_edge(pts):
        # Lateral offset: x position relative to center
        lateral_offset = pts[:, 0] - center_x
        # Normalize to [-1, 1] range representing +/- 10m
        lateral_offset_norm = lateral_offset / (img_width / 2.0)
        
        # Height: y position normalized (0=top of image, 1=bottom)
        # This matches how images are indexed: y=0 is top row
        height = pts[:, 1] / img_height
        
        return np.stack([lateral_offset_norm, height], axis=1)
    
    left_edge_norm = normalize_edge(left_edge_pts)
    right_edge_norm = normalize_edge(right_edge_pts)
    
    return left_edge_norm, right_edge_norm


def generate_edge_annotations(data_dir: str, output_json: str = None):
    """
    Generate edge annotations for all masks in dataset
    
    Args:
        data_dir: Directory containing images/ and masks/ folders
        output_json: Path to save annotations (default: data_dir/edge_annotations.json)
    """
    data_path = Path(data_dir)
    image_dir = data_path / 'images'
    mask_dir = data_path / 'masks'
    
    if output_json is None:
        output_json = data_path / 'edge_annotations.json'
    
    # Find all masks
    mask_files = sorted(list(mask_dir.glob('*.png')))
    
    if not mask_files:
        print(f"No masks found in {mask_dir}")
        return
    
    annotations = {}
    failed = 0
    
    print(f"Processing {len(mask_files)} masks...")
    
    for mask_path in tqdm(mask_files):
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed += 1
            continue
        
        # Extract edges
        left_edge, right_edge = extract_edge_polylines_from_mask(mask)
        
        if left_edge is None or right_edge is None:
            failed += 1
            continue
        
        # Store annotation
        frame_id = mask_path.stem
        annotations[frame_id] = {
            'left_edge': left_edge.tolist(),
            'right_edge': right_edge.tolist(),
            'mask_path': str(mask_path.relative_to(data_path)),
            'num_points': len(left_edge)
        }
    
    # Save annotations
    with open(output_json, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nGenerated {len(annotations)} edge annotations")
    print(f"Failed: {failed}")
    print(f"Saved to: {output_json}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate edge annotations from segmentation masks')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing images/ and masks/ folders')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON path (default: data_dir/edge_annotations.json)')
    
    args = parser.parse_args()
    
    generate_edge_annotations(args.data_dir, args.output)
