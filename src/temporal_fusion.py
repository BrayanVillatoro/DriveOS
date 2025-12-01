from collections import deque
import numpy as np


class TemporalFusion:
    """Simple temporal fusion for centerlines.

    Stores recent centerlines (Nx2 arrays) and produces an averaged centerline
    by resampling each to a fixed number of points along arc length and
    averaging the coordinates.
    """

    def __init__(self, maxlen=8):
        self.buf = deque(maxlen=maxlen)

    def add_centerline(self, pts):
        """Add a centerline (Nx2). Drops if too short."""
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 4:
            return False
        self.buf.append(pts)
        return True

    def clear(self):
        self.buf.clear()

    def count(self):
        return len(self.buf)

    def get_average_centerline(self, num=400):
        """Return averaged centerline resampled to `num` points, or empty array."""
        if len(self.buf) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # resample each centerline to `num` points using linear interpolation by cumulative distance
        resampled = []
        for pts in list(self.buf):
            d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
            u = np.concatenate(([0.0], np.cumsum(d)))
            if u[-1] <= 0:
                continue
            u = u / u[-1]
            uq = np.linspace(0.0, 1.0, num)
            xs = np.interp(uq, u, pts[:, 0])
            ys = np.interp(uq, u, pts[:, 1])
            resampled.append(np.vstack([xs, ys]).T)

        if not resampled:
            return np.empty((0, 2), dtype=np.float32)

        # average across buffers
        arr = np.stack(resampled, axis=0)  # (M, num, 2)
        mean = arr.mean(axis=0)
        return mean.astype(np.float32)
