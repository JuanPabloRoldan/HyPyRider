'''
==================================================
File: taylor_maccoll_solver.py
Purpose: Implements the oblique shock solver for compressible flow analysis.

Authors (listed alphabetically)
------------------------------------
Benjamin Lasher
Courtney Paternak
Dominic Perito
Juan P. Roldan

Last Updated: 4/22/2025

Nomenclature:
    gamma   : Specific heat ratio (dimensionless), default is 1.4 for air
    M1      : Freestream Mach number upstream of the shock
    theta_s : Shock wave angle (radians)
    delta   : Flow deflection angle (radians)
    M2      : Downstream Mach number
    Vr      : Radial velocity component (normalized)
    Vtheta  : Normal velocity component (normalized)
==================================================
'''

import numpy as np
import pandas as pd
from isentropic_relations_solver import IsentropicRelationsSolver

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, step_size=0.0001):
        '''
        Initializes the Taylor-Maccoll solver with default parameters.

        Parameters
        ----------
        gamma : float
            Specific heat ratio, default is 1.4 for air.
        step_size : float
            Integration step size in radians.
        '''
        self.gamma = gamma
        self.h = step_size
        self.gasConst = 278.15 
        self.theta_s = 17.2 #Degree
        self.M3 = 2.273     #This and the shock angle are set by user
        self.mach = 7
        self.gasConst = 287
        self.Temp2 = 293    #Maybe the correct temperature 2
        self.speed_sound = self.gamma*self.gasConst*self.Temp2


    def step_23(self):
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

        return Mn2, Mn3

    def step_456(self, Mn2, tol=1e-5, max_iter=100):
        """
        Steps 4 through 6 that iterate until delta converges.

        Parameters:
            Mn2 (float): Normal Mach number from step 3.
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum iterations allowed.

        Returns:
            delta:
            M2:
            Iterations:
        """
        # Step 4
        Delta = np.radians(5)  # 5 degrees as an initial guess
        diff = 1.0
        iteration = 0

        while diff > tol and iteration < max_iter:
            Beta = self.theta_s + Delta

            # Step 5
            print(type(Mn2))  # Should be a float or int
            print(type(Beta))  # Should be a float or int
            M2 = Mn2 / np.sin(Beta)

            # Step 6
            num = Mn2**2-1
            den = Mn2**2*(self.gamma+np.cos(2*Beta)+2)
            Delta_improved = np.arctan(2 * (1 / np.tan(Beta)) * (num / den))

            diff = abs((Delta_improved - Delta)/Delta_improved)
            Delta = Delta_improved
            iteration += 1

        return Delta, M2, iteration

    def Potentialmoccollfixer(self,Mn2, M2):
      """
      Uses updated euqations to obtain V_r and V_Theta

      Parameters:
          Mn2
          M2

      Returns:
          V theta
          V r
      """  
      V_theta = -Mn2*np.sqrt(self.gamma*self.gasConst*self.Temp2)
      V_r = (M2**2 - Mn2**2)**0.5 * np.sqrt(self.gamma*self.gasConst*self.Temp2)

      return V_theta, V_r

    def calculate_Mach_from_components(self, V_r, V_theta):
        '''
        Calculates the Mach number from the radial and tangential velocity components.

        Parameters
        ----------
        V_r : float
            Radial component of velocity (V'_r, normalized).
        V_theta : float
            Tangential component of velocity (V'_theta, normalized).

        Returns
        -------
        float
            M : float
                Computed Mach number.
        '''
        # Compute the normalized velocity magnitude (V')
        V_prime = np.sqrt(V_r ** 2 + V_theta ** 2)

        # Compute the inverse squared velocity magnitude term
        temp = (1 / (V_prime ** 2)) - 1
        print(type(temp))  # Should be a float or int
        print(temp)

        # Compute the Mach number (M) from the velocity magnitude
        M = np.sqrt((2 / temp) / (self.gamma - 1))

        # Return the computed Mach number
        return M

    def taylor_maccoll_system(self, theta, Vr, dVr):
        '''
        Defines the Taylor-Maccoll ODE system.

        Parameters
        ----------
        theta : float
            Angle of the position vector from the cone vertex (radians).
        Vr : float
            Radial component of velocity (normalized).
        dVr : float
            First derivative of Vr with respect to theta.

        Returns
        -------
        np.array
            A 2-element array containing dVr and ddVr.
        '''
        B = (self.gamma - 1) / 2 * (1 - Vr**2 - dVr**2)
        C = (2 * Vr + dVr / np.tan(theta))
        numerator = dVr**2 - (B * C)
        denominator = B - dVr**2
        ddVr = numerator / denominator
        return np.array([dVr, ddVr])

    def rk4_step(self, theta, Vr, dVr):
        '''
        Performs a single RK4 integration step for Taylor-Maccoll equations.

        Parameters
        ----------
        theta : float
            Current angle (radians).
        Vr : float
            Current radial velocity.
        dVr : float
            Current derivative of radial velocity.

        Returns
        -------
        tuple
            Updated values of Vr and dVr after one step.
        '''
        # K1 and M1
        K1, M1 = self.taylor_maccoll_system(theta, Vr, dVr)
        K1 = self.h * K1
        M1 = self.h * M1

        # K2 and M2
        K2 = self.h * (dVr + 0.5 * M1)
        M2 = self.h * self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * K1, 
            dVr + 0.5 * M1
        )[1]

        # K3 and M3
        K3 = self.h * (dVr + 0.5 * M2)
        M3 = self.h * self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * K2, 
            dVr + 0.5 * M2
        )[1]

        # K4 and M4
        K4 = self.h * (dVr + M3)
        M4 = self.h * self.taylor_maccoll_system(
            theta + self.h, 
            Vr + K3, 
            dVr + M3
        )[1]

        # Update Vr and dVr
        Vr_next = Vr + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        dVr_next = dVr + (1 / 6) * (M1 + 2 * M2 + 2 * M3 + M4)

        return Vr_next, dVr_next

    def solve(self, theta0, Vr0, dVr0):
        '''
        Solves the Taylor-Maccoll equation and returns the final values.

        Parameters
        ----------
        theta0 : float
            Initial angle (radians).
        Vr0 : float
            Initial radial velocity.
        dVr0 : float
            Initial derivative of Vr.

        Returns
        -------
        tuple
            cone angle (radians), V_r, and V_theta.
        '''
        theta = theta0
        Vr = Vr0
        dVr = -dVr0

        while abs(dVr) > 1e-3:  # Continue until abs(dVr/dtheta) < 1e-3
            # Perform RK4 step
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h

        # Return final values
        return theta, Vr, dVr
    
    def tabulate_from_shock_to_cone(self, theta_s, theta_c, Vr0, dVr0):
        '''
        Solves the Taylor-Maccoll equation and returns a DataFrame with results.

        Parameters
        ----------
        theta_s : float
            Shock angle (radians).
        theta_c : float
            Cone angle (radians).
        Vr0 : float
            Initial radial velocity.
        dVr0 : float
            Initial derivative of Vr.

        Returns
        -------
        pd.DataFrame
            DataFrame containing Theta (radians), V_r, and V_theta.
        '''
        isentropic_solver = IsentropicRelationsSolver(self.gamma)

        theta = theta_c
        Vr = Vr0
        dVr = dVr0
        M = self.calculate_Mach_from_components(Vr, dVr)

        isentropic_properties = isentropic_solver.isentropic_relations(M)
        p_ratio = isentropic_properties["Static Pressure Ratio (p/p0)"]
        t_ratio = isentropic_properties["Static Temperature Ratio (T/T0)"]
        rho_ratio = isentropic_properties["Static Density Ratio (rho/rho0)"]

        # Lists to store results
        results = [[theta, M, Vr, dVr, p_ratio, t_ratio, rho_ratio]]  # Log initial conditions

        while abs(theta - theta_s) > 1e-3:  # Continue until reach shock angle
            # Perform RK4 step
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            M = self.calculate_Mach_from_components(Vr, dVr)

            ######DELETE THE NEXT 3 LINES
            if np.isnan(M):
                print("⚠️ Mach number is NaN at this step")
                break

            isentropic_properties = isentropic_solver.isentropic_relations(M)
            p_ratio = isentropic_properties["Static Pressure Ratio (p/p0)"]
            t_ratio = isentropic_properties["Static Temperature Ratio (T/T0)"]
            rho_ratio = isentropic_properties["Static Density Ratio (rho/rho0)"]

            theta += self.h

            # Save current results
            results.append([theta, M, Vr, dVr, p_ratio, t_ratio, rho_ratio])

        # Create and return DataFrame
        results_df = pd.DataFrame(results, columns=["Theta (radians)", "Mach", "V_r", "V_theta", "P/P0", "T/T0", "rho/rho0"])
        return results_df

# Example usage
if __name__ == "__main__":
    solver = TaylorMaccollSolver()
    theta_s = np.radians(30)  # Example shock angle
    theta_c = np.radians(theta_s)
    Mc = 3.57846955
    M3 = 5

    Mn2, Mn3 = solver.step_23()
    M2, Delta, itterations = solver.step_456(Mn2)
    V_r, V_theta = solver.Potentialmoccollfixer(Mn2, M3)

    
    results_df = solver.tabulate_from_shock_to_cone(theta_s, theta_c, V_r, dVr0 = 1)
    print(results_df.head())
