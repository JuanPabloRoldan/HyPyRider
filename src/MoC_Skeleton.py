from method_of_characteristics import FlowProperties, Point, AxisymmetricMOC
from newton_raphson import newton_raphson_system
from NR_initial_guess import get_init_NR_guess
from moc_solver_pc import PredictorCorrectorExpansion

import numpy as np

class MoC_Skeleton:
    def __init__(self, Mach, wall_params, method="NR"):
        self.flow_props = FlowProperties()
        self.Mach = Mach
        self.wall_params = wall_params
        self.method = method.upper()

        if self.method == "NR":
            self.moc_solver = AxisymmetricMOC(wall_params, self.flow_props)

    def MoC_Mesher(self, leading_vertex, log_file="outputs/nr_debug_log.txt"):
        if self.method == "PC":
            print("Using Predictor-Corrector method for MoC mesh generation...")

            # Geometry preprocessing for PC
            x0, r0 = leading_vertex
            x1, x2 = self.wall_params["x1"], self.wall_params["x2"]
            r1, r2 = self.wall_params["r1"], self.wall_params["r2"]

            Z_par = np.linspace(x1, x2, 100).reshape(1, -1)
            X_par = np.linspace(r1, r2, 100).reshape(1, -1)
            Y_par = np.zeros_like(X_par)

            pc_solver = PredictorCorrectorExpansion(
                gamma=self.flow_props.gamma,
                initial_mach=self.Mach,
                X_par=X_par,
                Y_par=Y_par,
                Z_par=Z_par,
                x0=x0
            )

            return pc_solver.solve()

        print("Using Newton-Raphson method for MoC mesh generation...")
        i_max = 30
        delta_s = 0.1
        success = True

        moc_mesh = np.empty((i_max, i_max), dtype=object)
        x0, r0 = leading_vertex
        init_point = Point(x0, r0, 0, self.Mach, self.flow_props)
        moc_mesh[0][0] = init_point

        with open(log_file, "a") as log:
            log.write(f"Initial Point[0][0]: {init_point}\n")

        mu = init_point.mu

        for i in range(1, i_max):
            x_i = x0 + i * delta_s * np.cos(mu)
            r_i = r0 + i * delta_s * np.sin(mu)

            moc_mesh[i][0] = Point(x_i, r_i, 0.0, self.Mach, self.flow_props)
            with open(log_file, "a") as log:
                log.write(f"Point[{i}][0]: {moc_mesh[i][0]}\n")

            for j in range(1, i):
                P1 = moc_mesh[i][j - 1]
                P2 = moc_mesh[i - 1][j]

                if P1 is None or P2 is None:
                    success = False
                    with open(log_file, "a") as log:
                        log.write(f"ERROR: P1 or P2 is None at ({i},{j}). Halting mesh generation.\n")
                    return moc_mesh, False

                guess = get_init_NR_guess(P1, P2)

                solution = newton_raphson_system(
                    lambda v: self.moc_solver.evaluate(v, P1, P2),
                    lambda v: self.moc_solver.jacobian(v, P1, P2),
                    guess,
                    relaxation=0.8,
                    log_file=log_file
                )

                if solution is None:
                    success = False
                    with open(log_file, "a") as log:
                        log.write(f"FAILED: NR did not converge at point ({i},{j}). Halting mesh generation.\n\n")
                    return moc_mesh, False

                new_point = Point(x=solution[5], r=solution[4], theta=solution[0], M=solution[3], flow_props=self.flow_props)
                moc_mesh[i][j] = new_point
                with open(log_file, "a") as log:
                    log.write(f"Point[{i}][{j}]: {new_point}\n")

            # Wall point
            P1 = moc_mesh[i][i - 1]
            P2 = moc_mesh[i - 1][i - 1]

            if P1 is None or P2 is None:
                success = False
                with open(log_file, "a") as log:
                    log.write(f"ERROR: P1 or P2 is None at wall point ({i},{i}). Halting mesh generation.\n")
                return moc_mesh, False

            guess = get_init_NR_guess(P1, P2, is_wall=True)

            solution = newton_raphson_system(
                lambda v: self.moc_solver.evaluate(v, P1, P2, is_wall=True),
                lambda v: self.moc_solver.jacobian(v, P1, P2, is_wall=True),
                guess,
                relaxation=0.8,
                log_file=log_file
            )

            if solution is None:
                success = False
                with open(log_file, "a") as log:
                    log.write(f"FAILED: NR did not converge at wall point ({i},{i}). Halting mesh generation.\n\n")
                return moc_mesh, False

            new_point = Point(x=solution[5], r=solution[4], theta=solution[0], M=solution[3], flow_props=self.flow_props)
            moc_mesh[i][i] = new_point
            with open(log_file, "a") as log:
                log.write(f"Wall Point[{i}][{i}]: {new_point}\n")

            if moc_mesh[i][i].x >= 9:
                with open(log_file, "a") as log:
                    log.write(f"Terminating mesh generation: x >= 9 at i={i}\n")
                break

        return moc_mesh, success

if __name__ == "__main__":
    Mach_number = 10.0
    leading_vertex = [3.5010548, 3.5507]
    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}

    method = "PC"  # Change to "NR" to use Newton-Raphson solver
    moc_solver = MoC_Skeleton(Mach=Mach_number, wall_params=wall_params, method=method)
    result, success = moc_solver.MoC_Mesher(leading_vertex)

    if success:
        print("MoC mesh successfully generated!\n")

        if method == "PC":
            ZE, XE, TH, MU, Q, ME, NEP = result
            print("Shape of ZE:", ZE.shape)
        else:
            row_to_print = 5
            print(f"--- Mesh Row {row_to_print} ---")
            for j, pt in enumerate(result[row_to_print]):
                if pt is not None:
                    print(f"Point[{row_to_print}][{j}]: x={pt.x:.3f}, r={pt.r:.3f}, "
                          f"theta={np.degrees(pt.theta):.2f}°, M={pt.M:.3f}")
    else:
        print("MoC mesh generation encountered errors. Check log file.\n")