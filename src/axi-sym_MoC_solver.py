import os
from point import Point
from moc_solver_pc import AxisymMoC
import numpy as np
import matplotlib.pyplot as plt

class MoC_Skeleton:
    '''Builds a full axisymmetric MoC characteristic mesh, marching a grid of
    interior and wall points downstream from an initial point using
    AxisymMoC (see moc_solver_pc.py) for each individual point solve.'''

    def __init__(self, M_inf, a_inf, gamma, wall_params):
        '''
        Parameters
        ----------
        M_inf : float
            Freestream Mach number at the mesh's starting point.
        a_inf : float
            Freestream speed of sound (m/s), used to convert q_max to
            physical velocity units.
        gamma : float
            Specific heat ratio.
        wall_params : dict
            Parabolic wall geometry: {"x1", "r1", "x2", "r2"}, passed through
            to AxisymMoC.
        '''
        self.M_inf = M_inf
        self.a_inf = a_inf
        self.gamma = gamma
        self.wall_params = wall_params
        self.q_max = M_inf * a_inf * np.sqrt(1 + (2 / (gamma - 1)) * (1 / (M_inf * M_inf)))
        self.q_inf = self.q_max / np.sqrt(1 + (2 / (self.gamma - 1)) * (1 / M_inf ** 2))

        self.moc_solver = AxisymMoC(self.q_max, self.gamma, self.wall_params)

    def MoC_Mesher(self, log_file="src/outputs/nr_debug_log.txt"):
        '''
        Marches the characteristic mesh downstream from the initial point on
        the axis, row by row: each row starts with a new axis point, fills
        in interior points via solve_internal_point(), and closes with a
        wall point via solve_wall_point(). Stops early (returning the
        partial mesh) if either solve step fails to converge.

        Note: solve_wall_point() models the body as a parabola valid only
        within wall_params["x1"] to wall_params["x2"]. If i_max/delta_s are
        large enough that the mesh marches past x2 before reaching i_max
        rows, solve_wall_point() will fail once it tries to intersect a
        characteristic with the wall beyond that domain (see its docstring)
        -- this is a modeling-domain limit, not necessarily a bug. If you
        hit this, check whether wall_params covers the z-extent you
        actually need before assuming the math is wrong.

        Parameters
        ----------
        log_file : str
            Path to append a per-point debug log to.

        Returns
        -------
        np.ndarray of object
            i_max x i_max array of Point objects (or None for unfilled/
            unreached mesh cells).
        '''
        i_max = 50
        delta_s = 0.1

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        moc_mesh = np.empty((i_max, i_max), dtype=object)
        x0  = self.wall_params["x1"]
        r0 = self.wall_params["r1"]
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
            PC = self.moc_solver.solve_wall_point(PB)
            moc_mesh[i][i] = PC
            print(moc_mesh[i][i])
            if PC is None:
                return moc_mesh

        return moc_mesh

if __name__ == "__main__":
    M_inf = 7
    gamma = 1.2
    a_inf = np.sqrt(gamma * 287 * 231.64) # T_inf at 30,000 m

    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}

    moc_solver = MoC_Skeleton(M_inf, a_inf, gamma, wall_params)
    mesh = moc_solver.MoC_Mesher()

    z1 = wall_params["x1"]
    z2 = wall_params["x2"]
    r1 = wall_params["r1"]
    r2 = wall_params["r2"]

    # --- Collect all valid MoC points ---
    all_points = [pt for row in mesh for pt in row if pt is not None and not np.isnan(pt.x) and not np.isnan(pt.r)]
    x_vals = [pt.x for pt in all_points]
    r_vals = [pt.r for pt in all_points]

    # --- Define the parabolic surface function ---
    def r_b(z):
        return ((r2 - r1) / (z2 - z1)**2) * (z - z1)**2 + r1

    # --- Generate z and r values along the parabolic wall ---
    z_vals = np.linspace(z1, z2, 300)
    r_wall = r_b(z_vals)

    # --- Plotting ---
    plt.figure()
    plt.scatter(x_vals, r_vals, s=10, label="MoC Mesh Points")

    # Add thin black mesh lines
    for row in mesh:
        valid_row = [pt for pt in row if pt is not None and not np.isnan(pt.x) and not np.isnan(pt.r)]
        if len(valid_row) > 1:
            plt.plot([pt.x for pt in valid_row], [pt.r for pt in valid_row], color='black', linewidth=0.5)

    for col_idx in range(len(mesh[0])):
        col = [mesh[i][col_idx] for i in range(len(mesh)) if col_idx < len(mesh[i]) and mesh[i][col_idx] is not None]
        valid_col = [pt for pt in col if not np.isnan(pt.x) and not np.isnan(pt.r)]
        if len(valid_col) > 1:
            plt.plot([pt.x for pt in valid_col], [pt.r for pt in valid_col], color='black', linewidth=0.5)

    # Plot wall surface
    plt.plot(z_vals, r_wall, color='red', linewidth=2, label="Parabolic Wall Surface")

    plt.xlabel("x")
    plt.ylabel("r")
    plt.title("MoC Points with Parabolic Body Surface and Mesh Lines")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()