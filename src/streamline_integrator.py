import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer
import process_LE_points

class StreamlineIntegrator:
    def __init__(self, gamma, M1, theta_s):
        """
        Initializes the streamline integrator.
        This includes creating an instance of the conical flow solver,
        as well as saving a data frame of useful solutions from the conical shock
        solutions from theta = [shock angle, cone angle].

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
        self.TM_tabulation = self.conical_analyzer.tabulate_tm_shock_to_cone(theta_s)

    def trace_streamline(self, x, y, z):
        """
        Traces a streamline starting from the given leading-edge (LE) normalized coordinates.
        TODO: review procedure, and eventually have it such that 

        Parameters:
            x (float): Normalized x-coordinate of the LE point.
            y (float): Normalized y-coordinate of the LE point.
            z (float): Normalized z-coordinate of the LE point.

        Returns:
            TODO: Prints the evolution of the streamline to the console.
        """
        theta = self.theta_s
        theta = np.radians(theta)

        debug_inner_counter = 0

        while x < 0.5:
            debug_inner_counter += 1
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            alpha = np.arctan(abs(z / y))

            dt = 0.02

            # Interpolate V_r and V_theta from tabulated data
            V_r = np.interp(np.degrees(theta), self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_r'])
            V_theta = np.interp(np.degrees(theta), self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_theta'])
            print(f'Vr={V_r}\tVtheta={V_theta}')

            # Update theta and r
            d_theta = V_theta * dt / r
            theta += d_theta

            r += V_r * dt
            w = np.sqrt(y ** 2 + z ** 2)

            # Update coordinates
            x = r * np.cos(theta)
            y = w * np.cos(alpha)
            z = w * np.sin(alpha)

            print(f'theta={np.degrees(theta)}')
            print(f'x={x}\ty={y}\tz={z}')

    def create_lower_surface(self):
        """
        Generates the lower surface by tracing streamlines from the leading-edge (LE) points.

        The function normalizes the LE points, traces each streamline, and prints the 
        trajectory to the console.

        Parameters:
            None

        Returns:
            TODO
        """
        # Use the shock wave angle as the starting point for tracing
        theta_0 = self.theta_s

        # Find the maximum x-coordinate to normalize the lengths
        max_x = self.LE_points['X'].max()  # Assumes LE_points is a DataFrame with a column 'X'
        self.ref_length = max_x

        debug_counter = 0
        for index, row in self.LE_points.iterrows():
            x, y, z = row['X'], row['Y'], row['Z']
            x /= self.ref_length
            y /= self.ref_length
            z /= self.ref_length
            print(f'NEW POINT')
            print(f'x={x}\ty={y}\tz={z}')

            # Trace the streamline for the current LE point
            self.trace_streamline(x, y, z)
            print('\n\n\n')
            debug_counter += 1

# Example Usage
if __name__ == "__main__":
    # Initialize the streamline integrator with specific parameters
    integrator = StreamlineIntegrator(gamma=1.2, M1=10.0, theta_s=20)

    # Extract leading-edge points from a file
    file_path = 'src/inputs/LeadingEdgeData_LeftSide.nmb'
    integrator.LE_points = process_LE_points.extract_points_from_file(file_path)

    # Create the lower surface by tracing streamlines
    integrator.create_lower_surface()