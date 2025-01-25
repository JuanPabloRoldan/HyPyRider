import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer

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

        # Solve for cone angle using instance method
        self.theta_c, _, _ = self.conical_analyzer.solve_taylor_maccoll(theta_s)

    def tabulate_flowfield(self):
        """
        Tabulates the post-shock flow field properties over a range of angles [shock angle to cone angle].

        Returns:
            pd.DataFrame: A DataFrame containing shock angles, cone angles, V_r, and V_theta.
        """
        theta_range = np.arange(self.theta_s, self.theta_c, -0.1)
        
        # Use the instance to call solve_taylor_maccoll_range
        data = self.conical_analyzer.solve_taylor_maccoll_range(theta_range)
        
        print(theta_range)
        print(data)
        return data

# Example Usage
if __name__ == "__main__":
    integrator = StreamlineIntegrator(gamma=1.405, M1=10.0, theta_s=20)
    integrator.tabulate_flowfield()