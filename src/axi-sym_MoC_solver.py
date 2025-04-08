from point import Point
from moc_solver_pc import AxisymMoC
import numpy as np

class MoC_Skeleton:
    def __init__(self, M_inf, a_inf, gamma, wall_params):
        self.M_inf = M_inf
        self.a_inf = a_inf
        self.gamma = gamma
        self.wall_params = wall_params
        self.q_max = M_inf * a_inf * np.sqrt(1 + (2 / (gamma - 1)) * (1 / (M_inf * M_inf)))
        self.q_inf = self.q_max / np.sqrt(1 + (2 / (self.gamma - 1)) * (1 / M_inf ** 2))

        self.moc_solver = AxisymMoC(self.q_max, self.gamma, self.wall_params)

    def MoC_Mesher(self, log_file="outputs/nr_debug_log.txt"):
        i_max = 30
        delta_s = 0.1
        success = True

        moc_mesh = np.empty((i_max, i_max), dtype=object)
        x0  = wall_params["x1"]
        r0 = wall_params["r1"]
        init_point = Point(x0, r0, 0, self.M_inf, self.q_inf)
        print(init_point)
        moc_mesh[0][0] = init_point

        with open(log_file, "a") as log:
            log.write(f"Initial Point[0][0]: {init_point}\n")

        mu = init_point.mu

        for i in range(1, i_max):
            x_i = x0 + i * delta_s * np.cos(mu)
            r_i = r0 + i * delta_s * np.sin(mu)

            moc_mesh[i][0] = Point(x_i, r_i, 0.0, self.M_inf, self.q_inf)
            print(moc_mesh[i][0])
            with open(log_file, "a") as log:
                log.write(f"Point[{i}][0]: {moc_mesh[i][0]}\n")

            for j in range(1, i):
                PA = moc_mesh[i - 1][j]
                PB = moc_mesh[i][j - 1]
                PC = self.moc_solver.solve_internal_point(PA, PB)
                if PC is None:
                    return moc_mesh
                moc_mesh[i][j] = PC
                print(moc_mesh[i][j])

            # for a point, C, at the wall
            PB = moc_mesh[i][i - 1]
            print(PB)
            PC = self.moc_solver.solve_wall_point(PB)
            moc_mesh[i][i] = PC
            print(moc_mesh[i][i])
            if PC is None:
                return moc_mesh

        return moc_mesh

if __name__ == "__main__":
    M_inf = 10.0
    a_inf = 300
    gamma = 1.2
    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}

    moc_solver = MoC_Skeleton(M_inf, a_inf, gamma, wall_params)
    mesh = moc_solver.MoC_Mesher()