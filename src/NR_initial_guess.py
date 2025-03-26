# For the Newton Raphson method, we need the inital conditions. We are finding theta_3, mu_3, M3, x3, y3, and nu3. We are given
# theta_1, mu_1, M1, x1, y1, and nu1. We are also given theta_2, mu_2, M2, x2, y2, and nu2. 
# We are importing the known conditions from another file to use in this file.
import numpy as np
from method_of_characteristics import Point
from method_of_characteristics import FlowProperties

def get_init_NR_guess(p1, p2, is_wall=False):
    """
    Computes an initial guess for NR based on averages of two known flow Points.

    Parameters:
    p1, p2: Point
        Known flow points along the C- and C+ characteristics.

    Returns:
    numpy.ndarray
        Initial guess vector [theta3, nu3, mu3, M3, r3, x3]
    """

    theta2 = p2.theta
    nu2 = p2.nu
    mu2 = p2.mu
    M2 = p2.M
    r2 = p2.r
    x2 = p2.x

    if is_wall:

        flow_solver = FlowProperties()

        M3 = 1.05 * M2
        mu3 = flow_solver.mach_angle(M3)
        nu3 = flow_solver.prandtl_meyer(M3)
        theta3 = nu3 + nu2
        r3 = 0.98 * r2
        x3 = 1.05 * x2

        return np.array([theta3, nu3, mu3, M3, r3, x3])
    
    theta1 = p1.theta
    nu1 = p1.nu
    mu1 = p1.mu
    M1 = p1.M
    r1 = p1.r
    x1 = p1.x

    theta3 = 0.5 * (p1.theta + p2.theta)
    nu3    = 0.5 * (p1.nu + p2.nu)
    mu3    = 0.5 * (p1.mu + p2.mu)
    M3     = 0.5 * (p1.M  + p2.M)
    r3  = 0.5 * (p1.r  + p2.r)
    x3  = 0.5 * (p1.x  + p2.x)

    return np.array([theta3, nu3, mu3, M3, r3, x3])