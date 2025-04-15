import numpy as np
import taylor_maccoll_solver as tm
import streamline_integrator as si
import pandas as pd

class BusemannInlet:
    
    def __init__(self,mach,gamma,Temp2):
        """
        Initializes the busemann inlet class.
        """
        self.theta_s = 17.2 #Degree
        self.M3 = 2.273     #This and the shock angle are set by user
        self.gamma = gamma
        self.mach = mach
        self.gasConst = 287
        self.Temp2 = Temp2
        self.speed_sound = self.gamma*self.gasConst*self.Temp2

    def step2_3(self):
        """
        Performs steps 2 and 3 in the Busemann inlet procces

        Parameters:

        Returns:
            Mn2 = normal component of M2
            Mn3 = normal component of M3
        """
        #Step 2
        Mn3 = np.sin(self.theta_s)

        #Step 3
        num = Mn3**2+self.gamma/(self.gamma-1)
        den = Mn3**2*(2*self.gamma)/(self.gamma-1)-1
        Mn2 = np.sqrt(num/den)

        return{
            "Mn2" : Mn2,
            "Mn3" : Mn3
        }

    def step_456(self, Mn2, tol=1e-5, max_iter=100):
        """
        Steps 4 through 6 that iterate until delta converges.

        Parameters:
            Mn2 (float): Normal Mach number from step 3.
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum iterations allowed.

        Returns:
            dict: Contains converged delta and M2.
        """
        # Step 4
        Delta = np.radians(5)  # 5 degrees as an initial guess
        diff = 1.0
        iteration = 0

        while diff > tol and iteration < max_iter:
            Beta = self.theta_s + Delta

            # Step 5
            M2 = Mn2 / np.sin(Beta)

            # Step 6
            num = Mn2**2-1
            den = Mn2**2*(self.gamma+np.cos(2*Beta)+2)
            Delta_improved = np.arctan(2 * (1 / np.tan(Beta)) * (num / den))

            diff = abs(Delta_improved - Delta)
            Delta = Delta_improved
            iteration += 1

        return {
            "Delta": Delta,
            "M2": M2,
            "Iterations": iteration
        }

    def solve_conical_flow(self, M2, Delta,Mn2, Mn3):
        """
        Solves the conical flow field using the Taylor-Maccoll solver.

        Parameters:
            M2 (float): Downstream Mach number.
            Delta (float): Flow deflection angle.

        Returns:
            pd.DataFrame: Taylor-Maccoll flow field table.
        """
        solver = tm.TaylorMaccollSolver(gamma=self.gamma)
        theta_c = np.radians(self.theta_s)  # cone angle ≈ shock angle
        V_theta, V_r = solver.calculate_velocity_components(self,Mn2, Mn3)
        df = solver.tabulate_from_shock_to_cone(theta_s=self.theta_s, theta_c=theta_c, Vr0=Vr0, dVr0=dVr0)
        return df

    def Potentialmoccollfixer(self,Mn2, Mn3):  #####MAYBE DELETE LATER
        """
        I think fixes the taylor maccoll issue.
        TM is getting its own vr and Vtheta

        Parameters:
            Mn2
            Mn3

        Returns:
            V theta
            V r
        """  
        V_theta = -Mn2*np.sqrt(self.gamma*self.gasConst*self.Temp2)
        V_r = (Mn3**2 - Mn2**2)**0.5 * np.sqrt(self.gamma*self.gasConst*self.Temp2)

        return{
            "V_Theta" : V_theta,
            "Vr" : V_r
        }


if __name__ == "__main__":
    inlet = BusemannInlet(mach=3.0, gamma=1.4,Temp2=350.0)

    # Step 2-3
    results = inlet.step2_3()
    Mn2 = results["Mn2"]
    Mn3 = results["Mn3"]

    # Step 4-6
    step456_results = inlet.step_456(Mn2)
    delta = step456_results["Delta"]
    M2 = step456_results["M2"]

    # Taylor-Maccoll solver
    df = inlet.solve_conical_flow(M2, Mn2, Mn3) #delta removed
