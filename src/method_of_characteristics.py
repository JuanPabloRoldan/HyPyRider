import numpy as np
from newton_raphson import newton_raphson_system

class Point:
    """Class to store properties of a flow point."""
    def __init__(self, x, r, theta, M, flow_props):
        """
        Initializes a flow point with given parameters.
        
        Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        theta (float): Flow angle in radians.
        M (float): Mach number.
        flow_props (FlowProperties): Instance to compute flow properties.
        """
        self.x = x  # X-coordinate
        self.r = r  # Y-coordinate
        self.theta = theta  # Flow angle (radians)
        self.M = M  # Mach number

        # Compute flow-dependent properties
        self.mu = flow_props.mach_angle(M)  # Mach angle
        self.nu = flow_props.prandtl_meyer(M)  # Prandtl-Meyer function

    def __repr__(self):
        """String representation for debugging."""
        return (f"Point(x={self.x:.2f}, r={self.r:.2f}, "
                f"theta={np.degrees(self.theta):.2f}°, M={self.M:.3f}, "
                f"mu={np.degrees(self.mu):.2f}°, nu={np.degrees(self.nu):.2f}°)")

class FlowProperties:
    """Handles calculations related to flow properties."""
    def __init__(self, gamma=1.4):
        """Initialize the FlowProperties class."""
        self.gamma = gamma  # Store gamma in FlowProperties

    def mach_angle(self, M):
        """Computes the Mach angle for a given Mach number."""
        if M > 1:
            return np.arcsin(1 / M)  # Mach angle in radians
        else:
            raise ValueError("Mach angle is undefined for M <= 1")

    def prandtl_meyer(self, M):
        """Computes the Prandtl-Meyer function in radians."""
        if M > 1:
            term1 = np.sqrt((self.gamma + 1) / (self.gamma - 1))
            term2 = np.arctan(np.sqrt((self.gamma - 1) * (M**2 - 1) / (self.gamma + 1)))
            term3 = np.arctan(np.sqrt(M**2 - 1))
            return term1 * term2 - term3
        else:
            raise ValueError("Prandtl-Meyer function is undefined for M <= 1")

class AxisymmetricMOC:
    """Class to handle the Axisymmetric Method of Characteristics (MOC) calculations."""
    def __init__(self, flow_properties):
        """
        Initialize the AxisymmetricMOC solver.
        
        Parameters:
        flow_properties (FlowProperties): Instance of FlowProperties for calculations.
        """
        self.flow_properties = flow_properties  # Store reference to FlowProperties

    def system_equations(self, vars, point1, point2, is_wall=False, wall_params=None):
        """
        Defines the nonlinear system of equations for solving unknowns at Point 3.
        
        Parameters:
        vars (array): [x3, y3, theta3, M3, mu3, nu3] - unknowns at Point 3.
        point1 (Point): Known point for C- characteristic (if inner point).
        point2 (Point): Known point for C+ characteristic.
        is_wall (bool): Flag indicating whether the point is on a wall.
        wall_params (dict): Contains parameters for wall equations (if at a wall).
        
        Returns:
        array: Residuals of the equations.
        """
        theta3, nu3, mu3, M3, r3, x3 = vars

        # Compute nu3 and mu3 using given M3 (redundant, but useful for checking consistency)
        nu3_computed = self.flow_properties.prandtl_meyer(M3)
        mu3_computed = self.flow_properties.mach_angle(M3)

        if is_wall:
            # Extract wall parameters
            r1 = wall_params["r1"]
            r2 = wall_params["r2"]
            x1 = wall_params["x1"]

            # Wall Equation (Eq. 8)
            eq1 = r3 - ((r2 - r1) / (1 - x1)**2 * (x3 - x1)**2 + r1)

            # Wall Derivative Equation (Eq. 9)
            eq3 = (2 * (r2 - r1) / (1 - x1)**2) * (x3 - x1) - np.tan(theta3)

        else:
            # Standard characteristic equations (C-)
            eq1 = (r3 - point1.r) - (x3 - point1.x) * np.tan(0.5 * (point1.theta - point1.mu + theta3 - mu3))

            # Compatibility equation for C-
            eq3 = (theta3 - point1.theta + nu3 - point1.nu) - (
                (0.5 * (M3**2 + point1.M**2) - 1) ** 0.5 * 2 / np.cot(point1.theta + theta3) * (r3 - point1.r) / (r3 + point1.r)
            )

        # Characteristic equation for C+ (always the same)
        eq2 = (r3 - point2.r) - (x3 - point2.x) * np.tan(0.5 * (point2.theta + point2.mu + theta3 + mu3))

        # Compatibility equation for C+ (always the same)
        eq4 = (theta3 + point2.theta - nu3 + point2.nu) - (
            (0.5 * (M3**2 + point2.M**2) - 1) ** 0.5 * 2 / np.cot(point2.theta + theta3) * (r3 - point2.y) / (r3 + point2.r)
        )

        # Prandtl-Meyer Function (always the same)
        eq5 = nu3 - nu3_computed

        # Mach Angle Definition (always the same)
        eq6 = np.sin(mu3) - (1 / M3)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

    def evaluate(self, vars, point1, point2, is_wall=False, wall_params=None):
        """
        Wrapper for evaluating system residuals.
        """
        _, F = self.compute_jacobian(vars, point1, point2, is_wall, wall_params)
        return F

    def jacobian(self, vars, point1, point2, is_wall=False, wall_params=None):
        """
        Wrapper for evaluating the Jacobian matrix.
        """
        J, _ = self.compute_jacobian(vars, point1, point2, is_wall, wall_params)
        return J

    
    def compute_jacobian(self, vars, point1, point2, is_wall=False, wall_params=None):
        theta3, nu3, mu3, M3, r3, x3 = vars

        J = np.zeros((6, 6))
        F = np.zeros(6)

        J[0], F[0] = self._jacobian_c_minus_characteristic(vars, point1)
        J[1], F[1] = self._jacobian_c_plus_characteristic(vars, point2)
        J[2], F[2] = self._jacobian_c_minus_compatibility(vars, point1)
        J[3], F[3] = self._jacobian_c_plus_compatibility(vars, point2)
        J[4], F[4] = self._jacobian_mach_angle(vars)
        J[5], F[5] = self._jacobian_prandtl_meyer(vars)

        return J, F

    def _jacobian_c_minus_characteristic(self, vars, point):
        theta3, _, mu3, _, r3, x3 = vars
        dx = x3 - point.x
        dtheta = theta3 - mu3 + point.theta - point.mu
        tan_term = np.tan(0.5 * dtheta)

        J = np.zeros(6)
        J[0] = -0.5 / (np.cos(0.5 * dtheta) ** 2)  # dF/dtheta3
        J[1] = 0
        J[2] = 0.5 / (np.cos(0.5 * dtheta) ** 2)   # dF/dmu3
        J[3] = 0
        J[4] = 1
        J[5] = -tan_term

        F = r3 - point.r - dx * tan_term
        return J, F

    def _jacobian_c_plus_characteristic(self, vars, point):
        theta3, _, mu3, _, r3, x3 = vars
        dx = x3 - point.x
        dtheta = theta3 + mu3 + point.theta + point.mu
        tan_term = np.tan(0.5 * dtheta)

        J = np.zeros(6)
        J[0] = 0.5 / (np.cos(0.5 * dtheta) ** 2)   # dF/dtheta3
        J[1] = 0
        J[2] = 0.5 / (np.cos(0.5 * dtheta) ** 2)   # dF/dmu3
        J[3] = 0
        J[4] = 1
        J[5] = -tan_term

        F = r3 - point.r - dx * tan_term
        return J, F

    def _jacobian_c_minus_compatibility(self, vars, point):
        theta3, _, mu3, M3, r3, _ = vars
        C1 = theta3 + mu3 - (point.theta + point.mu)
        C2 = np.sqrt(0.5 * (M3 ** 2 + point.M ** 2) - 1)
        C3 = (r3 - point.r) / (r3 + point.r)
        C4 = 0.5 * (theta3 + point.theta)

        J = np.zeros(6)
        if C4 == 0:
            F = C1
            J[0] = 1
            J[1] = 1
            J[3] = 0
            J[4] = 0
        else:
            cotC4 = 1 / np.tan(C4)
            denom = C2 - cotC4
            F = C1 - 2 * C3 / denom
            dC4 = 0.5
            J[0] = 1 + 2 * C3 / (denom ** 2) * (1 / (np.sin(C4) ** 2)) * dC4
            J[1] = 1
            J[3] = C3 * M3 / (C2 * denom ** 2)
            J[4] = -4 * point.r / ((r3 + point.r) ** 2 * denom)
        return J, F

    def _jacobian_c_plus_compatibility(self, vars, point):
        theta3, nu3, _, M3, r3, _ = vars
        C1 = theta3 - nu3 - (point.theta - point.nu)
        C2 = np.sqrt(0.5 * (M3 ** 2 + point.M ** 2) - 1)
        C3 = (r3 - point.r) / (r3 + point.r)
        C4 = 0.5 * (theta3 + point.theta)

        J = np.zeros(6)
        if C4 == 0:
            F = C1
            J[0] = 1
            J[1] = -1
            J[3] = 0
            J[4] = 0
        else:
            cotC4 = 1 / np.tan(C4)
            denom = C2 + cotC4
            F = C1 - 2 * C3 / denom
            dC4 = 0.5
            J[0] = 1 - 2 * C3 / (denom ** 2) * (1 / (np.sin(C4) ** 2)) * dC4
            J[1] = -1
            J[3] = C3 * M3 / (C2 * denom ** 2)
            J[4] = -4 * point.r / ((r3 + point.r) ** 2 * denom)
        return J, F

    def _jacobian_mach_angle(self, vars):
        _, _, mu3, M3, _, _ = vars

        J = np.zeros(6)
        F = np.sin(mu3) - 1 / M3

        J[2] = np.cos(mu3)  # dF/dmu3
        J[3] = 1 / (M3 ** 2)  # dF/dM3
        return J, F

    def _jacobian_prandtl_meyer(self, vars):
        _, nu3, _, M3, _, _ = vars

        J = np.zeros(6)
        gamma = self.flow_properties.gamma
        C1 = (gamma + 1) / (gamma - 1)
        C2 = M3 ** 2 - 1
        sqrtC2 = np.sqrt(C2)
        sqrtC2_C1 = np.sqrt(C2 / C1)

        F = nu3 - np.sqrt(C1) * np.arctan(sqrtC2_C1) + np.arctan(sqrtC2)

        dF_dM = (M3 / sqrtC2) * (1 / (1 + C2) - 1 / (1 + (C2 / C1)))

        J[1] = 1     # dF/dnu3
        J[3] = dF_dM # dF/dM3
        return J, F

# ---- TESTING THE CLASS ---- #
if __name__ == "__main__":
    import numpy as np
    from newton_raphson import newton_raphson_system

    # Create an instance of FlowProperties
    flow_props = FlowProperties()

    # Create an instance of AxisymmetricMOC using the flow properties
    moc_solver = AxisymmetricMOC(flow_props)

    # Define two known points (now passing flow_props directly)
    point1 = Point(x=0.0, r=0.0, theta=np.radians(5), M=2.0, flow_props=flow_props)
    point2 = Point(x=1.0, r=1.0, theta=np.radians(7), M=2.5, flow_props=flow_props)

    # Define guess for unknowns: [theta3, nu3, mu3, M3, r3, x3]
    guess = np.array([np.radians(6.0), flow_props.prandtl_meyer(2.25), flow_props.mach_angle(2.25), 2.25, 0.5, 0.5])

    solution = newton_raphson_system(
    lambda v: moc_solver.evaluate(v, point1, point2),
    lambda v: moc_solver.jacobian(v, point1, point2),
    guess
    )

    # Print point properties
    print("\n--- Point Properties ---")
    print(point1)
    print(point2)

    # Print solution
    print("\n--- Solution Found ---")
    print("theta3 (deg):", np.degrees(solution[0]))
    print("nu3 (deg):", np.degrees(solution[1]))
    print("mu3 (deg):", np.degrees(solution[2]))
    print("M3:", solution[3])
    print("r3:", solution[4])
    print("x3:", solution[5])
