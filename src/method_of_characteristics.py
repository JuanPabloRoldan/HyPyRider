import numpy as np

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
        return (f"Point(x={self.x:.2f}, y={self.y:.2f}, "
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

    def compute_jacobian(self, vars, point1, point2, is_wall=False, wall_params=None):
        """
        Computes the Jacobian matrix (6x6) for the system equations.

        Parameters:
        vars (array): [x3, y3, theta3, M3, mu3, nu3] - unknowns at Point 3.
        point1 (Point): Known point for C- characteristic (if inner point).
        point2 (Point): Known point for C+ characteristic.
        is_wall (bool): Flag indicating whether the point is on a wall.
        wall_params (dict): Contains parameters for wall equations (if at a wall).

        Returns:
        np.array: The 6x6 Jacobian matrix.
        """
        theta3, nu3, mu3, M3, r3, x3 = vars

        J = np.zeros((6, 6))  # Initialize the 6x6 Jacobian matrix

        # Right running C- Characteristic Line
        # ====================================
        C1 = r3 - point1.r / (x3 - point1.x)
        C2 = (theta3 - mu3) + (point1.theta - point1.mu)
        F1 = C1 - np.tan(C2)

        J[0, 0] = -1 / (2 * np.cos(C2) ** 2)
        J[0, 1] = 0
        J[0, 2] = -J[0,0]
        J[0 , 3] = 0
        J[0, 4] = 1 / (x3 - point1.x)
        J[0, 5] = -C1 * J[0, 4]

        # Left running C+ Characteristic Line
        # ====================================
        C1 = r3 - point2.r / (x3 - point2.x)
        C2 = (theta3 + mu3) + (point2.theta + point2.mu)
        F2 = C1 - np.tan(C2)

        J[1, 0] = -1 / (2 * np.cos(C2) ** 2)
        J[1, 1] = 0
        J[1, 2] = J[1, 0]
        J[1, 3] = 0
        J[1, 4] = 1 / (x3 - point2.x)
        J[1, 5] = -C1 * J[1, 4]
        
        # Right running C- Compatibility Eq.
        # ====================================
        C1 = (theta3 + mu3) - (point1.theta + point1.mu)
        C2 = np.sqrt(-1 + 0.5 * (M3 * M3 + point1.M * point1.M))
        C3 = (r3 - point1.r) / (r3 + point1.r)
        C4 = 0.5 * (theta3 + point1.theta)
        F3 = C1 - (2 * C3) / (C2 - (1 / np.tan(C4)))

        J[2, 0] = 1 + C3 / (np.sin(C2) ** 2 * (C2 - (1 / np.tan(C4))) ** 2)
        J[2, 1] = 1
        J[2, 2] = 0
        J[2, 3] = C3 * M3 / (C2 * (C2 - (1 / np.tan(C4))) ** 2)
        J[2, 4] = -4 * point1.r / ((r3 + point1.r) ** 2 * (C2 - (1 / np.tan(C4))))
        J[2, 5] = 0

# ---- TESTING THE CLASS ---- #
if __name__ == "__main__":
    # Create an instance of FlowProperties
    flow_props = FlowProperties()

    # Create an instance of AxisymmetricMOC using the flow properties
    moc_solver = AxisymmetricMOC(flow_props)

    # Define two known points (now passing flow_props directly)
    point1 = Point(x=0.0, y=0.0, theta=np.radians(5), M=2.0, flow_props=flow_props)
    point2 = Point(x=1.0, y=1.0, theta=np.radians(7), M=2.5, flow_props=flow_props)

    # Solve for the intersection (x3, y3)
    # x3, y3 = moc_solver.find_intersection(point1, point2)

    # Print point properties
    print("\n--- Point Properties ---")
    print(point1)
    print(point2)

    # Print characteristic slopes (already stored in Point)
    print("\n--- Characteristic Slopes ---")
    print(f"Slope of C+ at Point 1: {point1.slope_C_plus:.4f}")
    print(f"Slope of C- at Point 1: {point1.slope_C_minus:.4f}")
    print(f"Slope of C+ at Point 2: {point2.slope_C_plus:.4f}")
    print(f"Slope of C- at Point 2: {point2.slope_C_minus:.4f}")

    # Print compatibility equations (already stored in Point)
    print("\n--- Compatibility Equations ---")
    print(f"Compatibility C+ at Point 1: {np.degrees(point1.compatibility_C_plus):.2f} degrees")
    print(f"Compatibility C- at Point 1: {np.degrees(point1.compatibility_C_minus):.2f} degrees")
    print(f"Compatibility C+ at Point 2: {np.degrees(point2.compatibility_C_plus):.2f} degrees")
    print(f"Compatibility C- at Point 2: {np.degrees(point2.compatibility_C_minus):.2f} degrees")

    # Print intersection
    print("\n--- Intersection Point ---")
    # print(f"Intersection at x3={x3:.4f}, y3={y3:.4f}")