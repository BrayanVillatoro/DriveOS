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
    
    # Openpilot-style: Use distance ahead (X_IDXS) instead of pixel heights
    # X_IDXS with quadratic spacing: denser near car, sparser far ahead
    max_distance = 50.0  # meters ahead visible in racing
    X_IDXS = np.array([max_distance * ((i / (num_points - 1))**2) for i in range(num_points)])
    
    def sample_edge_at_distances(pts, distances, img_h):
        """Sample edge points at specific distances (converted from y pixel rows)"""
        # Convert y pixels to approximate distance
        # Assume perspective: bottom of image = 0m, top = max_distance
        y_coords = pts[:, 1]
        x_coords = pts[:, 0]
        
        # Map y pixels to distance (0=bottom/near, img_h=top/far)
        pt_distances = max_distance * (1.0 - y_coords / img_h)
        
        # Interpolate x coordinates at desired distances
        sampled_x = np.interp(distances, pt_distances[::-1], x_coords[::-1])
        return sampled_x
    
    left_x = sample_edge_at_distances(left_pts, X_IDXS, img_height)
    right_x = sample_edge_at_distances(right_pts, X_IDXS, img_height)
    
    # Convert to openpilot format: (X, Y) coordinates
    # X = distance ahead (already have as X_IDXS)
    # Y = lateral offset from center in meters
    
    def normalize_edge(x_coords):
        # Lateral offset: x position relative to center
        lateral_offset = x_coords - center_x
        # Convert pixels to meters (assume ~3.7m road width = image width)
        road_width = 10.0  # meters
        lateral_offset_m = lateral_offset * (road_width / img_width)
        
        # Stack: [distance_ahead, lateral_offset]
        return np.stack([X_IDXS, lateral_offset_m], axis=1)
    
    left_edge_norm = normalize_edge(left_x)
    right_edge_norm = normalize_edge(right_x)
    
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
