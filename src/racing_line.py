import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class RacingLineEstimator:
    def __init__(self, vehicle_params=None):
        self.vehicle = vehicle_params or {
            "mu": 1.1,
            "g": 9.81,
            "max_accel": 6.0,
            "max_brake": 8.0,
        }

    def extract_centerline_from_video(self, video_path):
        raise NotImplementedError(
            "Centerline extraction is project-specific. Plug your own method here (ML segmentation / CSV / manual annotations)."
        )

    def extract_centerline_from_segmentation(self, seg_map: np.ndarray, class_track: int = 0,
                                             sample_every_n_rows: int = 6, min_width: int = 10,
                                             bias_to_inner: bool = True, bias_strength: float = 0.35) -> np.ndarray:
        """
        Extract a centerline (Nx2 pixel coordinates) from a segmentation map.

        Strategy (simple, robust, fast):
        - seg_map: HxW array with integer labels (0 = track surface)
        - For each row (y) sampled at 'sample_every_n_rows' find the largest continuous track region
          and use its midpoint as the center x coordinate.
        - Filter rows with too-narrow regions and perform a final smoothing using a cubic spline
          fit across the sampled points.

        Returns: numpy array of shape (M, 2) in pixel coordinates (x, y)
        """
        # ensure ints
        seg = seg_map.astype(np.int32)
        h, w = seg.shape[:2]

        # binary track mask
        mask = (seg == class_track).astype(np.uint8)
        if mask.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Find the largest connected track component to remove off-track islands
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return np.empty((0, 2), dtype=np.float32)

        # choose the largest component except background (label 0)
        # stats[:, cv2.CC_STAT_AREA] gives areas
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        comp_mask = (labels == largest_idx).astype(np.uint8)

        points = []
        widths = []
        ys = np.arange(0, h, sample_every_n_rows)
        for y in ys:
            row = comp_mask[y, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0:
                continue
            if xs.size < min_width:
                continue
            # center by mean to cope with non-symmetric shapes
            center_x = float(xs.mean())
            width = float(xs.max() - xs.min())
            points.append((center_x, float(y)))
            widths.append(width)

        if len(points) < 4:
            # not enough points for spline smoothing, return raw points
            return np.array(points, dtype=np.float32)

        pts = np.array(points)
        widths = np.array(widths) if len(widths) == len(pts) else np.full(len(pts), 0.0)
        # smooth with low-order spline to remove jitter
        try:
            from scipy.interpolate import splprep, splev
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=2.0)
            u_fine = np.linspace(0, 1, max(200, len(pts) * 10))
            x_s, y_s = splev(u_fine, tck)
            smooth = np.vstack([x_s, y_s]).T.astype(np.float32)
            if not bias_to_inner:
                return smooth

            # Estimate turn direction locally and bias center towards inner edge
            # Compute curvature sign from spline derivatives
            dx, dy = splev(u_fine, tck, der=1)
            ddx, ddy = splev(u_fine, tck, der=2)
            curv_signed = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)
            # guard against NaNs/Infs
            curv_signed = np.nan_to_num(curv_signed, nan=0.0, posinf=0.0, neginf=0.0)

            # Map each original sampled y to closest u index
            # Then compute biased center using half-width and curvature sign
            biased_pts = []
            # Bound widths to reasonable range to avoid huge shifts from bad masks
            widths = np.clip(widths, 0.0, max(w * 0.9, 1.0))
            half_widths = widths * 0.5
            for (cx, yy), hw in zip(pts, half_widths):
                # nearest index in y_s
                idx = int(np.argmin(np.abs(y_s - yy)))
                turn_sign = np.sign(curv_signed[idx])  # + left, - right in image coordinates
                # If width is very small or no clear turn, skip bias
                if hw < 4.0 or turn_sign == 0.0:
                    bx = cx
                else:
                    # shift center towards inner edge by bias_strength * half_width, capped
                    max_shift = hw * 0.6
                    shift = np.clip(bias_strength * hw, 0.0, max_shift)
                    bx = cx - turn_sign * shift
                # keep inside image
                bx = float(np.clip(bx, 0.0, float(w - 1)))
                biased_pts.append((bx, yy))

            biased_pts = np.array(biased_pts, dtype=np.float32)
            # re-spline biased points for a smooth final centerline
            try:
                tck_b, _ = splprep([biased_pts[:, 0], biased_pts[:, 1]], s=2.0)
                x_b, y_b = splev(np.linspace(0, 1, len(u_fine)), tck_b)
                return np.vstack([x_b, y_b]).T.astype(np.float32)
            except Exception:
                # fallback to lightly smoothed polyline
                return cv2.GaussianBlur(biased_pts.astype(np.float32), (1, 1), 0)
        except Exception:
            return pts.astype(np.float32)

    def fit_spline(self, centerline_pts, smoothing=0.0):
        centerline_pts = np.array(centerline_pts)
        x, y = centerline_pts[:, 0], centerline_pts[:, 1]
        tck, u = splprep([x, y], s=smoothing)
        return tck, u

    def compute_curvature_and_arc_length(self, tck, num=2000):
        u_eval = np.linspace(0, 1, num)
        x, y = splev(u_eval, tck)
        dx, dy = splev(u_eval, tck, der=1)
        ddx, ddy = splev(u_eval, tck, der=2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
        ds = np.sqrt(dx**2 + dy**2)
        arc = np.cumsum(ds)
        return (x, y), curvature, arc

    def compute_speed_profile(self, curvature, arc):
        mu = self.vehicle["mu"]
        g = self.vehicle["g"]
        v_lat = np.sqrt(np.maximum(mu * g / (curvature + 1e-8), 0))
        v_fwd = np.copy(v_lat)
        for i in range(1, len(v_fwd)):
            dv = self.vehicle["max_accel"] * (arc[i] - arc[i - 1])
            v_fwd[i] = min(v_lat[i], np.sqrt(v_fwd[i - 1] ** 2 + 2 * dv))
        v_bwd = np.copy(v_lat)
        for i in range(len(v_bwd) - 2, -1, -1):
            dv = self.vehicle["max_brake"] * (arc[i + 1] - arc[i])
            v_bwd[i] = min(v_bwd[i], np.sqrt(v_bwd[i + 1] ** 2 + 2 * dv))
        speed = np.minimum(v_fwd, v_bwd)
        return speed

    def overlay_racing_line(self, frame, spline, speed_profile, thickness=3):
        x, y = spline
        pts = np.vstack([x, y]).T.astype(np.float32)
        # filter invalid / nan points
        if pts.size == 0:
            return frame
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] < 2:
            return frame
        h, w = frame.shape[:2]
        # clamp to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        # remove nearly-duplicate consecutive points
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate(([True], diffs > 1.0))
        pts = pts[keep]
        if pts.shape[0] < 2:
            return frame
        # Resample speed_profile to match number of points so color mapping is correct
        speed_arr = np.asarray(speed_profile)
        if speed_arr.ndim == 0:
            speed_arr = np.repeat(speed_arr, len(pts))
        if len(speed_arr) != len(pts):
            # interpolate speed to match pts length
            speed_resampled = np.interp(np.linspace(0, len(speed_arr) - 1, len(pts)), np.arange(len(speed_arr)), speed_arr)
        else:
            speed_resampled = speed_arr

        norm_speed = (speed_resampled - speed_resampled.min()) / (speed_resampled.max() - speed_resampled.min() + 1e-6)
        pts_int = np.round(pts).astype(np.int32)
        for i in range(len(pts_int) - 1):
            color = (
                int(255 * norm_speed[i]),
                0,
                int(255 * (1 - norm_speed[i])),
            )
            p0 = (int(pts_int[i, 0]), int(pts_int[i, 1]))
            p1 = (int(pts_int[i + 1, 0]), int(pts_int[i + 1, 1]))
            cv2.line(frame, p0, p1, color, thickness)
        return frame

    def estimate_lap_time(self, arc, speed_profile):
        dt = np.diff(arc) / (speed_profile[:-1] + 1e-6)
        return dt.sum()

    def optimize_racing_line(self, centerline, n_control=8, maxiter=200, mu=None, g=None):
        """
        Optimize a lateral offset (in same units as centerline) applied across a small number
        of control points to reduce lap time. This is a practical, lightweight approach:
        - Sample centerline into dense curve
        - Pick n_control knot positions and optimize lateral offsets (signed) normal to local tangent
        - For each candidate offsets, shift the dense spline laterally and compute lap time

        Returns: optimal dense spline (x,y), curvature, arc, speed
        NOTE: This is a heuristic optimizer (Nelder-Mead) intended for batch/offline use.
        """
        try:
            from scipy.optimize import minimize
        except Exception:
            raise RuntimeError("scipy.optimize required for optimize_racing_line")

        # Use vehicle params
        mu = mu if mu is not None else self.vehicle['mu']
        g = g if g is not None else self.vehicle['g']

        centerline = np.array(centerline)
        # Fit coarse spline and sample it
        tck, _ = self.fit_spline(centerline, smoothing=0.0)
        dense_u = np.linspace(0, 1, 800)
        cx, cy = splev(dense_u, tck)
        dx, dy = splev(dense_u, tck, der=1)

        # compute normals (unit)
        tang = np.vstack([dx, dy]).T
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        nx = -tang[:, 1] / norms[:, 0]  # left-hand normal
        ny = tang[:, 0] / norms[:, 0]

        # choose control u positions
        ctrl_us = np.linspace(0, 1, n_control)
        ctrl_idx = np.searchsorted(dense_u, ctrl_us)

        # initial offsets = zeros
        x0 = np.zeros(n_control, dtype=float)

        def lap_time_for_offsets(offsets):
            # create smooth lateral offset along dense samples by linear interpolation
            offsets = np.asarray(offsets)
            offset_interp = np.interp(np.linspace(0, 1, len(dense_u)), ctrl_us, offsets)
            sx = cx + nx * offset_interp
            sy = cy + ny * offset_interp

            # fit spline back and compute curvature/arc
            try:
                tck2, _ = splprep([sx, sy], s=0)
            except Exception:
                return 1e9
            (sx_s, sy_s), curvature, arc = self.compute_curvature_and_arc_length(tck2, num=len(dense_u))
            # lateral-limited speed
            v_lat = np.sqrt(np.maximum(mu * g / (np.abs(curvature) + 1e-8), 0))
            # acceleration/brake passes
            speed = self.compute_speed_profile(curvature, arc)
            # lap time
            lap = self.estimate_lap_time(arc, speed)
            return lap

        res = minimize(lap_time_for_offsets, x0, method='Nelder-Mead', options={'maxiter': maxiter, 'disp': False})

        # construct final optimized spline
        best_offsets = res.x
        offset_interp = np.interp(np.linspace(0, 1, len(dense_u)), ctrl_us, best_offsets)
        sx = cx + nx * offset_interp
        sy = cy + ny * offset_interp
        tck3, _ = splprep([sx, sy], s=0)
        spline, curvature, arc = self.compute_curvature_and_arc_length(tck3, num=len(dense_u))
        speed = self.compute_speed_profile(curvature, arc)

        return (spline, curvature, arc, speed, res)


def plot_debug_track(spline, curvature, speed):
    x, y = spline
    fig, axs = plt.subplots(3, 1, figsize=(8, 14))
    axs[0].plot(x, y)
    axs[0].set_title("Racing Line Spline")
    axs[1].plot(curvature)
    axs[1].set_title("Curvature")
    axs[2].plot(speed)
    axs[2].set_title("Speed Profile")
    plt.tight_layout()
    plt.show()
