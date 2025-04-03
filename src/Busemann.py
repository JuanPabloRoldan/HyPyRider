import numpty as np

class BusemannInlet:
    
    def __init__(self,mach,gamma):
        """
        Initializes the busemann inlet class.
        """
        self.mach = mach
        self.theta_s = 17.2 #Degree
        self.M3 = 2.273     #This and the shock angle are set by user
        self.gamma = gamma

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
        num = np.sqrt(Mn3*(self.gamma-1)+2)
        den = np.sqrt(2*Mn3-1(self.gamma-1))
        Mn2 = num/den

        return{
            "Mn2" : Mn2,
            "Mn3" : Mn3
        }

    def iterator_eqn(self,Mn2,Mn3):
        """
        Steps 4 through 6 that must be itterated

        Parameters:

        Returns:
        """
        #Step 4
        Delta_geuss = Delta_geuss
        Beta = self.theta_s+Delta_geuss

        #Step 5
        M2 = Mn2/np.sin(Beta)

        #Step 6
        num = Mn2**2-1
        den = Mn2**2(self.gamma+np.cos(2*Beta)**2+2)
        Delta_improved = 2*np.cot(Beta)(num/den)

        return{
            "Delta" : Delta_improved
        }
