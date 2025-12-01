import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def fit_spline(centerline, smoothing=0):
    """
    Fit a cubic B-spline to the track centerline.
    centerline: np.ndarray of shape (N, 2)
    Returns: tuple (tck, u)
    """
    x, y = centerline[:, 0], centerline[:, 1]
    tck, u = splprep([x, y], s=smoothing)
    return tck, u

def compute_curvature(tck, u):
    """
    Compute curvature along the spline.
    Returns: curvature array (N,)
    """
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    return curvature

def compute_speed_profile(curvature, mu=1.2, g=9.81, a_acc=6.0, a_brake=-10.0, ds=1.0):
    """
    Compute speed profile using lateral and longitudinal limits.
    Returns: speed array (N,)
    """
    v_lat = np.sqrt(np.abs(mu * g / (curvature + 1e-6)))
    N = len(curvature)
    v = np.copy(v_lat)
    # Forward pass (acceleration)
    for i in range(N - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_acc * ds))
    # Backward pass (braking)
    for i in range(N - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * abs(a_brake) * ds))
    return v

def plot_line(centerline, tck, u, speed):
    """
    Plot the racing line colored by speed and the speed profile.
    """
    x, y = splev(u, tck)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sc = plt.scatter(x, y, c=speed, cmap='viridis', s=5)
    plt.plot(centerline[:, 0], centerline[:, 1], 'k--', alpha=0.5, label='Centerline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Racing Line Colored by Speed')
    plt.colorbar(sc, label='Speed (m/s)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(speed)), speed)
    plt.xlabel('Arc Length Index')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Profile')
    plt.tight_layout()
    plt.show()

# Example usage:
# centerline = np.array([[0,0],[10,0],[20,5],[30,15],[40,30],[50,50]])
# tck, u = fit_spline(centerline)
# curvature = compute_curvature(tck, u)
# speed = compute_speed_profile(curvature)
# plot_line(centerline, tck, u, speed)
