import numpy as np
from typing import Optional


def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be Nx2 array")
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    return np.hstack([pts, ones])


def from_homogeneous(pts_h: np.ndarray) -> np.ndarray:
    pts_h = np.asarray(pts_h, dtype=np.float64)
    w = pts_h[:, 2:3]
    w[w == 0] = 1e-12
    return pts_h[:, :2] / w


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography H to Nx2 points (image->world or world->image)."""
    H = np.asarray(H, dtype=np.float64)
    pts_h = to_homogeneous(pts)
    mapped = (H @ pts_h.T).T
    return from_homogeneous(mapped)


def invert_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return np.linalg.inv(H)
