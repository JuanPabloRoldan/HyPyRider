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
    def __init__(self, wall_params, flow_properties):
        """
        Initialize the AxisymmetricMOC solver.
        
        Parameters:
        flow_properties (FlowProperties): Instance of FlowProperties for calculations.
        """
        self.wall_params = wall_params
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

    def evaluate(self, vars, point1, point2, is_wall=False, wall_params=None, log_file='outputs/nr_debug_log.txt'):
        """
        Wrapper for evaluating system residuals.
        """
        _, F = self.compute_jacobian(vars, point1, point2, is_wall, wall_params, log_file)
        return F

    def jacobian(self, vars, point1, point2, is_wall=False, wall_params=None, log_file='outputs/nr_debug_log.txt'):
        """
        Wrapper for evaluating the Jacobian matrix.
        """
        J, _ = self.compute_jacobian(vars, point1, point2, is_wall, wall_params, log_file)
        return J

    
    def compute_jacobian(self, vars, point1, point2, is_wall=False, wall_params=None, log_file='outputs/nr_debug_log.txt'):
        theta3, nu3, mu3, M3, r3, x3 = vars

        J = np.zeros((6, 6))
        F = np.zeros(6)

        try:
            J[0], F[0] = self._jacobian_c_minus_characteristic(vars, point1)
            J[4], F[4] = self._jacobian_mach_angle(vars)
            J[5], F[5] = self._jacobian_prandtl_meyer(vars)

            print("Jacobian row for f[4] (r):", J[4])
            print("Jacobian row for f[5] (x):", J[5])


            if is_wall:
                wall = self.wall_params if wall_params is None else wall_params
                J[1], F[1] = self._jacobian_c_plus_characteristic_wall(vars, wall)
                J[2], F[2] = self._jacobian_c_minus_compatibility_wall(vars, point2)
                J[3], F[3] = self._jacobian_c_plus_compatibility_wall(vars, wall)
            else:
                J[1], F[1] = self._jacobian_c_plus_characteristic(vars, point2)
                J[2], F[2] = self._jacobian_c_minus_compatibility(vars, point1)
                J[3], F[3] = self._jacobian_c_plus_compatibility(vars, point2)

        except Exception as e:
            with open(log_file, "a") as log:
                log.write(f"\nERROR in compute_jacobian: {str(e)}\n")
                log.write(f"vars = {vars}\n\n")
            raise e

        with open(log_file, "a") as log:
            log.write("--- compute_jacobian ---\n")
            log.write(f"is_wall = {is_wall}\n")
            log.write(f"vars = {vars}\n")
            for i, val in enumerate(F):
                log.write(f"F[{i}] = {val}\n")
            log.write("-------------------------\n\n")

        return J, F

    def _jacobian_c_plus_characteristic_wall(self, vars, wall_params):
        "Change to be a function of the expansion cylinder"
        theta3, nu3, mu3, M3, r3, x3 = vars
        x1 = wall_params["x1"]
        x2 = wall_params["x2"]
        r1 = wall_params["r1"]
        r2 = wall_params["r2"]

        J = np.zeros(6)
        # J[0] = 0
        # J[1] = 0
        # J[2] = 0
        # J[3] = 0
        J[4] = 1
        J[5] = -2 * (r2 - r1) / (x2 - x1)**2 * (x3 - x1)

        F = r3 - r1 - (r2 - r1) / (x2 - x1)**2 * (x3 - x1)**2 
        return J, F
    
    def _jacobian_c_minus_compatibility_wall(self, vars, point):
        "theta3 = nu3 - nu2"
        theta3, nu3, mu3, M3, r3, x3 = vars

        J = np.zeros(6)
        J[0] = -1
        J[1] = 1
        # J[2] = 0
        # J[3] = 0
        # J[4] = 0
        # J[5] = 0

        F = nu3 - point.nu - theta3
        return J, F
    
    def _jacobian_c_plus_compatibility_wall(self, vars, wall_params):
        "Change to be a function of the expansion cylinder"
        theta3, nu3, mu3, M3, r3, x3 = vars
        x1 = wall_params["x1"]
        x2 = wall_params["x2"]
        r1 = wall_params["r1"]
        r2 = wall_params["r2"]

        J = np.zeros(6)
        J[0] = 1/np.cos(theta3)**2
        # J[1] = 0
        # J[2] = 0
        # J[3] = 0
        # J[4] = 0
        J[5] = -2 * (r2 - r1) / (x2 - x1)**2

        F = np.tan(theta3)- (2*(r2 - r1) / (x2 - x1)**2 * (x3 - x1)**2)
        return J, F

    def _jacobian_c_minus_characteristic(self, vars, point):
        theta3, mu3, r3, x3 = vars
        C1 = (r2-point.r)/(x3 - point.x)
        C2 = 0.5 * (theta3 - mu3 + point.theta - point.mu)

        J = np.zeros(6)
        J[0] = -0.5 * (1 / np.cos(C2)**2) # ∂f₁/∂θ₃
        # J[1] = 0
        J[2] = 0.5 * (1 / np.cos(C2)**2) # ∂f₁/∂μ₃
        # J[3] = 0
        J[4] = 1 / (x3 - point.x) # ∂f₂/∂r₃
        J[5] = -C1 * (1 / (x3 - point.x)) # ∂f₂/∂x₃

        F = C1 - np.tan(C2)
        return J, F

    def _jacobian_c_plus_characteristic(self, vars, point):
        theta3, mu3, r3, x3 = vars
        C1 = (r3 - point.r) / (x3 - point.x)
        C2 = 0.5 * (point.theta + point.mu + theta3 + mu3)

        J = np.zeros(6)
        J[0] = -0.5 * (1 / np.cos(C2)**2) # ∂f₂/∂θ₃
        # J[1] = 0
        J[2] = -0.5 * (1 / np.cos(C2)**2) # ∂f₂/∂μ₃
        # J[3] = 0
        J[4] = 1 / (x3 - point.x) # ∂f₂/∂r₃
        J[5] = -C1 * (1 / (x3 - point.x)) # ∂f₂/∂x₃

        F = C1 - np.tan(C2)
        return J, F

    def _jacobian_c_minus_compatibility(self, vars, point):
        theta3, nu3, M3, r3 = vars
        C1 = (theta3 + nu3) - (point.theta + point.nu)
        C2 = np.sqrt(0.5 * (M3**2 + point.M**2) - 1)
        C3 = (r3 - point.r) / (r3 + point.r)
        C4 = 0.5 * (theta3 + point.theta)

        J = np.zeros(6)
        if abs(C4) < 1e-6:
            J[0] = 1 + C3
            J[1] = 1
            # J[2] = 0
            # J[3] = 0
            # J[4] = 0
            # J[5] = 0
            
            F = C1
        else:
            J[0] = 1 + (C3 / ((np.sin(C4)**2) * ((C2 - np.cot(C4))**2)))
            J[1] = 1
            # J[2] = 0
            J[3] = (C3 * M3)/(C2 * ((C2 - np.cot(C4))**2))
            J[4] = (-4 * point.r) / ((r3 + point.r)**2 * (C2 - np.cot(C4)))
            # J[5] = 0

            F = C1 - ((2 * C3) / (C2 - np.cot(C4)))
        return J, F

    def _jacobian_c_plus_compatibility(self, vars, point):
        theta3, nu3, M3, r3 = vars
        C1 = (theta3 - nu3) - (point.theta - point.nu)
        C2 = np.sqrt(0.5 * (M3**2 + point.M**2) - 1)
        C3 = (r3 - point.r) / (r3 + point.r)
        C4 = 0.5 * (theta3 + point.theta)

        # C4 --> 0, 1/tanC4 --> explodes, and 1/sin2C4 --> near singular J
        J = np.zeros(6)
        if abs(C4) < 1e-6:
            J[0] = 1 - C3 
            J[1] = -1
            # J[2] = 0
            # J[3] = 0
            # J[4] = 0
            # J[5] = 0

            F = C1
        else:
            J[0] = 1 - ((C3)/((np.sin(C4)**2) * ((C2 + np.cot(C4))**2)))
            J[1] = -1
            # J[2] = 0
            J[3] = (C3 * M3)/(C2 * ((C2 + np.cot(C4))**2))
            J[4] = -(4 * point.r)/(((r3 + point.r)**2) * (C2 + np.cot(C4)))
            # J[5] = 0

            F = C1 + ((2 * C3)/(C2 + np.cot(C4)))
        return J, F

    def _jacobian_mach_angle(self, vars):
        mu3, M3 = vars

        J = np.zeros(6)
        # J[0] = 0
        # J[1] = 0
        J[2] = np.cos(mu3)  # dF/dmu3
        J[3] = 1 / (M3 ** 2)  # dF/dM3
        # J[4] = 0
        # J[5] = 0
        
        F = np.sin(mu3) - 1 / M3
        return J, F

    def _jacobian_prandtl_meyer(self, vars):
        nu3, M3 = vars
        gamma = self.flow_properties.gamma
        C1 = (gamma + 1) / (gamma - 1)
        C2 = (M3 ** 2) - 1

        J = np.zeros(6)
        if M3 <= 1.01: # if M3 is very close to 1, sqrtC2 --> 0, derivative explodes
            dF_dM = 1e6

        else:
            dF_dM = (M3 / sqrtC2) * ((1 / (1 + C2)) - (1 / (1 + (C2 / C1))))

        # J[0] = 0
        J[1] = 1     # dF/dnu3
        # J[2] = 0
        J[3] = dF_dM # dF/dM3
        # J[4] = 0
        # J[5] = 0

        F = nu3 - (np.sqrt(C1) * np.arctan(np.sqrt(C2/C1))) + (np.arctan(np.sqrt(C2)))
        return J, F

# ---- TESTING THE CLASS ---- #
if __name__ == "__main__":
    import numpy as np
    from newton_raphson import newton_raphson_system

    # Create an instance of FlowProperties
    flow_props = FlowProperties()

    # Define wall parameters
    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": -3.5507, "r2": -2.5}

    # Create an instance of AxisymmetricMOC using the flow properties
    moc_solver = AxisymmetricMOC(wall_params, flow_props)

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
