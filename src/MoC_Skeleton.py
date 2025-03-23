import method_of_characteristics
import newton_raphson
import NR_initial_guess


class MoC_Skeleton:
    
    def __init__(self):
        """
        Initializes the MoC Skeleton class.
        """
        
        self.method_of_characteristics = method_of_characteristics()
        self.newton_raphson = newton_raphson()
        self.NR_inital_guess = NR_initial_guess()

    def MoC_Skeleton():
        """
        Skeletal outline of the method of characteristics process. Insert functions and methods as needed 
        
        Parameters:
            #TODO Add any parameters

        Returns:
            #TODO Add any return values
        """

        while asociated_x_value <= x_value_endofsurface: #stops at the end of the surface
            while i <= j: 
                #for every point j before and at the wall along a line i
                #TODO Insert solver to obatin point 3

                if i == j:          #Check to see if you are at a wall
                    is_wall = True  # Set a boolean to True
                    #TODO Function to obtain the x value ascociated with the final (i,j) point

                j += 1 #work down each i line untill the wall
            
            i += 1     #move to next i line

