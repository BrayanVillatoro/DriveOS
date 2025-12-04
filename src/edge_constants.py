"""
Edge detection constants and utilities following openpilot's approach
"""
import numpy as np

def index_function(idx, max_val=192, max_idx=32):
    """Quadratic spacing for distance points - denser near vehicle"""
    return max_val * ((idx/max_idx)**2)

class EdgeConstants:
    """Constants for edge detection following openpilot"""
    
    # Number of prediction points along each edge
    IDX_N = 33
    
    # Distance indices (0-192 meters ahead, quadratic spacing)
    X_IDXS = np.array([index_function(idx, max_val=192.0) for idx in range(IDX_N)])
    
    # Lateral offset range (meters from center)
    MAX_LATERAL_OFFSET = 10.0  # +/- 10m from camera center
    
    # Number of road edges (left, right)
    NUM_ROAD_EDGES = 2
    
    # Edge output dimensions
    EDGE_WIDTH = 2  # (lateral_offset, height)
    
    # Confidence thresholds
    EDGE_CONF_THRESHOLD = 0.3
