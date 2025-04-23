# point.py
import numpy as np

class Point:
    def __init__(self, x, r, theta, M, q):
        self.x = x
        self.r = r
        self.theta = theta
        self.M = M
        self.mu = np.arcsin(1 / M)  # local Mach angle
        self.q = q

    def __repr__(self):
        return (f"Point(x={self.x:.2f}, r={self.r:.2f}, "
                f"theta={np.degrees(self.theta):.2f}°, M={self.M:.3f}, "
                f"mu={np.degrees(self.mu):.2f}°, q={self.q:.2f})")