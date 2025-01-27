import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer
import process_LE_points

class StreamlineIntegrator:
    def __init__(self, gamma, M1, theta_s):
        """
        Initializes the streamline integrator.

        Parameters:
            gamma (float): Specific heat ratio for the fluid.
            M1 (float): Freestream Mach number upstream of the shock.
            theta_s (float): Shock wave angle in degrees.
        """
        self.gamma = gamma
        self.M1 = M1
        self.theta_s = theta_s

        # Create an instance of ConicalFlowAnalyzer
        self.conical_analyzer = ConicalFlowAnalyzer(M1, gamma)

        # Tabulate post-shock flow properties from shock angle to the cone angle
        # TODO - currently, "full" only analyzes from theta wedge to theta cone, we need up to theta shock
        self.TM_tabulation = self.conical_analyzer.solve_taylor_maccoll_full(theta_s)
        print(self.TM_tabulation)

    def trace_streamline(self, x, y, z):
        """
        Traces a streamline given the initial LE, normalized coordinates.

        Parameters:
            x (float): normalized x-coordinate of LE point
            y (float): normalized y-coordinate of LE point
            z (float): normalized z-coordinate of LE point
        """
        theta = self.theta_s
        theta = np.radians(theta)

        while x < 1:
            r = np.sqrt(x ** 2 + y ** 2 + z **2)
            alpha = np.arctan(abs(z / y))

            dt = 0.02

            # Interpolate V_r and V_theta
            V_r = np.interp(np.degrees(theta), self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_r'])
            V_theta = np.interp(np.degrees(theta), self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_theta'])
            print(f'Vr={V_r}\tVtheta={V_theta}')

            d_theta = V_theta * dt / r
            theta += d_theta

            r += V_r * dt
            w = np.sqrt(y ** 2 + z **2)

            x = r * np.cos(theta)
            y = w * np.cos(alpha)
            z = w * np.sin(alpha)

            print(f'theta={np.degrees(theta)}')
            print(f'x={x}\ty={y}\tz={z}')

    def create_lower_surface(self):
        """
        """
        theta_0 = self.theta_s

        # Find the maximum value of x to normalize lengths
        max_x = self.LE_points['X'].max()  # Assumes LE_points is a DataFrame with a column 'X'
        self.ref_length = max_x

        debug_counter = 0
        for index, row in self.LE_points.iterrows():
            if debug_counter == 1:
                break
            x, y, z = row['X'], row['Y'], row['Z']
            x /= self.ref_length
            y /= self.ref_length
            z /= self.ref_length
            print(f'NEW POINT')
            print(f'x={x}\ty={y}\tz={z}')

            self.trace_streamline(x, y, z)
            print('\n\n\n')
            debug_counter += 1


# Example Usage
if __name__ == "__main__":
    
    # Tabulate values of conical shocks properties
    integrator = StreamlineIntegrator(gamma=1.2, M1=10.0, theta_s=20)

    # Grab leading edge points
    file_path = 'src/inputs/LeadingEdgeData_LeftSide.nmb'
    integrator.LE_points = process_LE_points.extract_points_from_file(file_path)

    integrator.create_lower_surface()