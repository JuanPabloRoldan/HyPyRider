import numpy as np


class Point:
    '''Shared state object for a single characteristic mesh point, used
    throughout the axisymmetric MoC solver (moc_solver_pc.py,
    axi-sym_MoC_solver.py).'''

    def __init__(self, x, r, theta, M, q):
        '''
        Parameters
        ----------
        x : float
            Axial coordinate.
        r : float
            Radial coordinate.
        theta : float
            Flow angle relative to the axis of symmetry (radians).
        M : float
            Local Mach number.
        q : float
            Local velocity magnitude.
        '''
        self.x = x
        self.r = r
        self.theta = theta
        self.M = M
        self.mu = np.arcsin(1 / M)  # local Mach angle (radians)
        self.q = q

    def __repr__(self):
        return (f"Point(x={self.x:.2f}, r={self.r:.2f}, "
                f"theta={np.degrees(self.theta):.2f}°, M={self.M:.3f}, "
                f"mu={np.degrees(self.mu):.2f}°, q={self.q:.2f})")
