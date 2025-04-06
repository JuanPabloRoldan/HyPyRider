import numpy as np
import taylor_maccoll_solver as tm

class BusemannInlet:
    
    def __init__(self,mach,gamma):
        """
        Initializes the busemann inlet class.
        """
        self.theta_s = 17.2 #Degree
        self.M3 = 2.273     #This and the shock angle are set by user
        self.gamma = gamma
        self.mach = mach

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

    def iterator_eqn(self,Mn2):
        """
        Steps 4 through 6 that must be itterated

        Parameters:

        Returns:
            Delta = delta

        """
        #Step 4
        Delta_geuss = Delta_geuss
        Beta = self.theta_s+Delta_geuss

        #Step 5
        M2 = Mn2/np.sin(Beta)

        #Step 6
        num = Mn2**2-1
        den = Mn2**2(self.gamma+np.cos(2*Beta)+2)
        Delta_improved = np.arctan(2*(1/np.tan)*(Beta)(num/den))

        return{
            "Delta" : Delta_improved,
            "M2" : M2
        }
    
