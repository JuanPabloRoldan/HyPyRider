from method_of_characteristics import FlowProperties, Point, AxisymmetricMOC
import newton_raphson
import NR_initial_guess

import numpy as np


class MoC_Skeleton:
    
    def __init__(self, Mach):
        """
        Initializes the MoC Skeleton class.
        """
        self.flow_props = FlowProperties()
        self.moc_solver = AxisymmetricMOC(self.flow_props)
        self.newton_raphson = newton_raphson()
        self.NR_initial_guess = NR_initial_guess()

        self.Mach = Mach

    def MoC_Mesher(self, leading_vertex):
        """
        Skeletal outline of the method of characteristics process. Insert functions and methods as needed 
        
        Parameters:
            #TODO Add any parameters

        Returns:
            #TODO Add any return values
        """

        i_max = 30
        delta_s = 0.1

        # initialize square matrix to hold mesh
        moc_mesh = np.zeros(i_max, i_max)
        
        x0, r0 = leading_vertex
        init_point = Point(x0, r0, 0, self.Mach, self.flow_props)
        moc_mesh[0][0] = init_point

        mu = init_point.mu

        for i in range(1, i_max):
            # passing i as some i * delta | grab known point on the Mach cone
            x_i = x0 + i * delta_s * np.cos(mu)
            r_i = r0 + i * delta_s * np.sin(mu)

            moc_mesh[i][0] = cone_point = Point(x_i, r_i, 0.0, self.M, self.flow_props)

            for j in range(1, i): 
                # for every internal point j along a line i

                J, F = moc_solver.compute_jacobian(moc_mesh[i][j-1], moc_mesh[i-1][j])
                
            # once broken out of j loop, necessarily at a wall (ie j == 1)
            # moc_mesh[i][i] = solve_moc(moc_mesh[i][i-1], is_wall=True)

            # if moc_mesh[i][i].x >= cone_length
            #     break

if __name__ == "__main__":
    Mach_number = 10.0  # example input

    leading_vertex = [3.5010548, 3.5507]

    moc_solver = MoC_Skeleton(Mach=Mach_number)
    mesh = moc_solver.MoC_Mesher(leading_vertex)

    # Basic print to confirm structure
    print("MoC mesh initialized. Example corner entry:", mesh[0][0])