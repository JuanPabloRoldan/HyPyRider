from method_of_characteristics import FlowProperties, Point, AxisymmetricMOC
from newton_raphson import newton_raphson_system
from NR_initial_guess import get_init_NR_guess

import numpy as np


class MoC_Skeleton:
    
    def __init__(self, Mach):
        """
        Initializes the MoC Skeleton class.
        """
        self.flow_props = FlowProperties()
        self.moc_solver = AxisymmetricMOC(self.flow_props)
        # self.newton_raphson = newton_raphson()
        # self.NR_initial_guess = NR_initial_guess()

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
                guess = get_init_NR_guess(P1, P2)

                # Use Newton-Raphson solver
                solution = newton_raphson_system(
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
            guess = get_init_NR_guess(P1, P2, is_wall=True)

            solution = newton_raphson_system(
                lambda v: self.moc_solver.evaluate(v, P1, P2, is_wall=True),
                lambda v: self.moc_solver.jacobian(v, P1, P2, is_wall=True),
                guess
                )

            new_point = Point(
                x=solution[5], r=solution[4],
                theta=solution[0], M=solution[3],
                flow_props=self.flow_props
                )

            moc_mesh[i][i] = new_point

            if moc_mesh[i][i].x >= 9:
                break

        return moc_mesh

if __name__ == "__main__":
    Mach_number = 10.0  # Example freestream Mach number
    leading_vertex = [3.5010548, 3.5507]  # Initial point (x0, r0)

    moc_solver = MoC_Skeleton(Mach=Mach_number)
    mesh = moc_solver.MoC_Mesher(leading_vertex)

    print("MoC mesh successfully generated!\n")

    # Print a sample row of points to verify:
    row_to_print = 5
    print(f"--- Mesh Row {row_to_print} ---")
    for j, pt in enumerate(mesh[row_to_print]):
        if pt is not None:
            print(f"Point[{row_to_print}][{j}]: x={pt.x:.3f}, r={pt.r:.3f}, "
                  f"θ={np.degrees(pt.theta):.2f}°, M={pt.M:.3f}")