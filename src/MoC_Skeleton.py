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
        i_max = 30
        delta_s = 0.1

        moc_mesh = np.empty((i_max, i_max), dtype=object)

        x0, r0 = leading_vertex
        init_point = Point(x0, r0, 0, self.Mach, self.flow_props)
        moc_mesh[0][0] = init_point

        mu = init_point.mu

        for i in range(1, i_max):
            x_i = x0 + i * delta_s * np.cos(mu)
            r_i = r0 + i * delta_s * np.sin(mu)

            moc_mesh[i][0] = Point(x_i, r_i, 0.0, self.Mach, self.flow_props)

            for j in range(1, i):
                P1 = moc_mesh[i][j - 1]
                P2 = moc_mesh[i - 1][j]
                guess = self.NR_initial_guess.get_guess(P1, P2)

                # Use Newton-Raphson solver here (example only):
                solution = newton_raphson.newton_raphson_system(
                lambda v: self.moc_solver.evaluate(v, P1, P2),
                lambda v: self.moc_solver.jacobian(v, P1, P2),
                guess
                )

                new_point = Point(
                    x=solution[5], r=solution[4],
                    theta=solution[0], M=solution[3],
                    flow_props=self.flow_props
                )

                moc_mesh[i][j] = new_point

            # Wall point
            P1 = moc_mesh[i][i - 1]
            P2 = moc_mesh[i - 1][i - 1]
            moc_mesh[i][i] = self.moc_solver.compute_wall_point(P1, P2)

        return moc_mesh

if __name__ == "__main__":
    Mach_number = 10.0  # example input

    leading_vertex = [3.5010548, 3.5507]

    moc_solver = MoC_Skeleton(Mach=Mach_number)
    mesh = moc_solver.MoC_Mesher(leading_vertex)

    # Basic print to confirm structure
    print("MoC mesh initialized. Example corner entry:", mesh[0][0])