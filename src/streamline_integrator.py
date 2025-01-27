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
        self.TM_tabulation = self.conical_analyzer.solve_taylor_maccoll(theta_s)
        print(self.TM_tabulation)

    def trace_streamline(self, x, y, z):

        theta = self.theta_s

        while x < 9:
            r = np.sqrt(x ** 2 + y ** 2 + z **2)
            r /= 9.3969262078590852
            alpha = np.arctan(abs(z / y))

            dt = 0.02

            # Interpolate V_r and V_theta
            V_r = np.interp(theta, self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_r'])
            V_theta = np.interp(theta, self.TM_tabulation['Theta (degrees)'], self.TM_tabulation['V_theta'])
            print(f'Vr={V_r}\tVtheta={V_theta}')

            d_theta = V_theta * dt / r
            theta = np.radians(theta)
            theta += d_theta

            r += V_r * dt
            w = np.sqrt(y ** 2 + z **2)

            x = r * np.cos(theta)
            y = w * np.cos(alpha)
            z = w * np.sin(alpha)

            print(f'theta={theta}')
            print(f'x={x}\ty={y}\tz={z}')

    def create_lower_surface(self):
        """
        """
        theta_0 = self.theta_s

        for index, row in self.LE_points.iterrows():
            x, y, z = row['X'], row['Y'], row['Z']
            print(f'NEW POINT')
            print(f'x={x}\ty={y}\tz={z}')

            self.trace_streamline(x, y, z)
            print('\n\n\n')


# Example Usage
if __name__ == "__main__":
    
    # Tabulate values of conical shocks properties
    integrator = StreamlineIntegrator(gamma=1.2, M1=10.0, theta_s=20)

    # Grab leading edge points
    file_path = 'src/inputs/LeadingEdgeData_LeftSide.nmb'
    integrator.LE_points = process_LE_points.extract_points_from_file(file_path)

    integrator.create_lower_surface()