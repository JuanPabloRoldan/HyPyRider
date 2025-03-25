from method_of_characteristics import *
import newton_raphson
import NR_initial_guess

import numpy as np


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

        i_max = 30

        # initialize square matrix to hold mesh
        moc_mesh = np.zeroes(i_max, i_max)
        
        # init_point = # grab known point on the leading vertex
        # moc_mesh[0][0] = init_point

        for i in range(1, i_max):

            # cone_point = # passing i as some i * delta | grab known point on the Mach cone
            # moc_mesh[i][0] = cone_point

            for j in range(1, i): 
                # for every internal point j along a line i

                # moc_mesh[i][j] = solve_moc(moc_mesh[i][j-1].right, moc_mesh[i-1][j].left)
    
            # once broken out of j loop, necessarily at a wall (ie j == 1)
            # moc_mesh[i][i] = solve_moc(moc_mesh[i][i-1].right, is_wall=True)

            # if moc_mesh[i][i].x >= cone_length
            #     break