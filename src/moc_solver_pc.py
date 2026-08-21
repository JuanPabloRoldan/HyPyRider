'''
==================================================
File: moc_solver_pc.py
Purpose: Predictor-corrector axisymmetric Method of Characteristics (MoC)
solver. Equation numbers cited in comments (e.g. "Equation 2.31a") refer to
Bowcutt's 1986 PhD dissertation, "Optimization of Hypersonic Waveriders
Derived from Cone Flows Including Viscous Effects" (Univ. of Maryland).

Nomenclature:
    gamma    : Specific heat ratio (dimensionless)
    q_max    : Maximum (limiting) velocity magnitude
    theta    : Flow angle relative to the axis of symmetry (radians)
    mu       : Local Mach angle, mu = asin(1/M) (radians)
    q        : Local velocity magnitude
    z, r     : Axisymmetric coordinates (axial, radial)
==================================================
'''

import numpy as np

from point import Point


class AxisymMoC:
    '''Predictor-corrector solver for one step of the axisymmetric MoC mesh:
    given two known characteristic points, solves for the downstream point
    where their characteristics intersect (interior) or meet the body wall.'''

    def __init__(self, q_max: float, gamma: float, wall_params: dict[str, float]) -> None:
        '''
        Parameters
        ----------
        q_max : float
            Maximum (limiting) velocity magnitude for the flow.
        gamma : float
            Specific heat ratio.
        wall_params : dict
            Parabolic wall geometry: {"x1", "r1", "x2", "r2"} (see
            Equation 2.36 for how the wall radius is defined from these).
        '''
        self.gamma = gamma
        self.q_max = q_max
        self.wall_params = wall_params

    def solve_internal_point(
        self, PA: Point, PB: Point, max_iters: int = 15, tol: float = 1e-7
    ) -> Point:
        '''
        Solves for a new characteristic mesh point C at the intersection of
        the C+ characteristic through PA and the C- characteristic through PB,
        using a predictor-corrector iteration (Bowcutt eqs. 2.26-2.35).

        Parameters
        ----------
        PA : Point
            Point on the C+ characteristic (left-running).
        PB : Point
            Point on the C- characteristic (right-running).
        max_iters : int
            Maximum corrector iterations before giving up on convergence.
        tol : float
            Relative convergence tolerance on q_c between corrector steps.

        Returns
        -------
        Point
            The solved interior point C.
        '''

        z_a = PA.x
        r_a = PA.r
        theta_a = PA.theta
        M_a = PA.M
        mu_a = np.arcsin(1 / M_a)
        q_a = PA.q

        z_b = PB.x
        r_b = PB.r
        theta_b = PB.theta
        M_b = PB.M
        mu_b = np.arcsin(1 / M_b)
        q_b = PB.q

        # ******************************************************
        # Predictor Step
        # ******************************************************

        # Equations 2.31a and 2.31b
        drdz_a = np.tan(theta_a + mu_a)
        drdz_b = np.tan(theta_b - mu_b)

        # Equations 2.32a and 2.32b
        z_c_prime = ((r_b - r_a) + (z_a * drdz_a) - (z_b * drdz_b))/(drdz_a - drdz_b)
        r_c_prime = drdz_a * (z_c_prime - z_a) + r_a

        # Equation 2.29a
        denom_q = ((1 / (np.tan(mu_a))) / q_a) + ((1 / (np.tan(mu_b))) / q_b)
        part1 = theta_b - theta_a
        part2 = (1 / np.tan(mu_a)) + (1 / np.tan(mu_b))
        part3 = ((np.sin(theta_a) * np.sin(mu_a)) / (r_a * np.cos(theta_a + mu_a)) )* (z_c_prime - z_a)
        part4 = ((np.sin(theta_b) * np.sin(mu_b)) / (r_b * np.cos(theta_b - mu_b)) )* (z_c_prime - z_b)

        q_c_prime = (1 / denom_q) * (part1 + part2 + part3 + part4)

        # Equation 2.29b
        theta_c_prime = (
            theta_a
            + (1 / (q_a * np.tan(mu_a))) * (q_c_prime - q_a)
            - (np.sin(mu_a) * np.sin(theta_a) / (r_a * np.cos(theta_a + mu_a))) * (z_c_prime - z_a)
        )

        # Equation 2.27
        M_c_prime = np.sqrt(2 / ((self.gamma - 1) * (((self.q_max / q_c_prime) ** 2) - 1)))

        # Equation 2.26
        mu_c_prime = np.arcsin(1 / M_c_prime)

        # ******************************************************
        # Corrector Step (iterative)
        # ******************************************************
        for _ in range(max_iters):
            # Equations 2.35a and 2.35b
            drdz_a = 0.5 * (np.tan(theta_a + mu_a) + np.tan(theta_c_prime + mu_c_prime))
            drdz_b = 0.5 * (np.tan(theta_b - mu_b) + np.tan(theta_c_prime - mu_c_prime))

            # Equation 2.32a and 2.32b
            z_c_new = ((r_b - r_a) + (z_a * drdz_a) - (z_b * drdz_b))/(drdz_a -drdz_b)
            r_c_new = drdz_a * (z_c_new - z_a) + r_a

            C1 = 0.5 * ((1 / (np.tan(mu_c_prime) * q_c_prime)) + (1 / (np.tan(mu_a) * q_a)))

            C2_a = np.sin(mu_a) * np.sin(theta_a) / (r_a * np.cos(theta_a + mu_a))
            C2_b = np.sin(mu_c_prime) * np.sin(theta_c_prime) / (r_c_prime * np.cos(theta_c_prime + mu_c_prime))
            C2 = 0.5 * (C2_a + C2_b) * (z_c_new - z_a)

            C3 = 0.5 * ((1 / (np.tan(mu_c_prime) * q_c_prime)) + (1 / (np.tan(mu_b) * q_b)))
            C4_a = np.sin(mu_b) * np.sin(theta_b) / (r_b * np.cos(theta_b - mu_b))
            C4_b = np.sin(mu_c_prime) * np.sin(theta_c_prime) / (r_c_prime * np.cos(theta_c_prime - mu_c_prime))
            C4 = 0.5 * (C4_a + C4_b) * (z_c_new - z_b)

            # Equations 2.33a and 2.33b
            q_c_new = (C2 + C4 + (theta_b - theta_a) + (C1 * q_a) + (C3 * q_b)) / (C1 + C3)
            theta_c_new = theta_a + C1 * (q_c_new - q_a) - C2

            # Equation 2.27
            M_c_new = np.sqrt(2 / ((self.gamma - 1) * (((self.q_max / q_c_new) ** 2) - 1)))

            # Equation 2.26
            mu_c_new = np.arcsin(1 / M_c_new)

            if abs(q_c_new - q_c_prime) / q_c_prime < tol:

                break

            z_c_prime, r_c_prime, theta_c_prime, M_c_prime, mu_c_prime, q_c_prime = \
                z_c_new, r_c_new, theta_c_new, M_c_new, mu_c_new, q_c_new

        return Point(z_c_prime, r_c_prime, theta_c_prime, M_c_prime, q_c_prime)

    def solve_wall_point(
        self, PB: Point, max_iters: int = 15, tol: float = 1e-7
    ) -> Point | None:
        '''
        Solves for a new characteristic mesh point C on the parabolic body
        wall, where the C- characteristic through PB meets the wall,
        using a predictor-corrector iteration (Bowcutt eqs. 2.36-2.41).

        Parameters
        ----------
        PB : Point
            Point on the C- characteristic (right-running) approaching the wall.
        max_iters : int
            Maximum corrector iterations before giving up on convergence.
        tol : float
            Relative convergence tolerance on q_c between corrector steps.

        Returns
        -------
        Point or None
            The solved wall point C, or None if the predictor step's
            quadratic (Equation 2.38) has no real solution (det < 0).
        '''
        z_b = PB.x
        r_b = PB.r
        theta_b = PB.theta
        M_b = PB.M
        mu_b = np.arcsin(1 / M_b)
        q_b = PB.q

        z1 = self.wall_params["x1"]
        z2 = self.wall_params["x2"]
        r1 = self.wall_params["r1"]
        r2 = self.wall_params["r2"]

        # ******************************************************
        # Predictor Step
        # ******************************************************

        # Equation 2.31b
        drdz_b = np.tan(theta_b - mu_b)

        # Equation 2.38
        a = drdz_b
        b = (r2 - r1) / ((z2 - z1) ** 2)
        c = b * (z1 * z1) + (a * z_b) - r_b +r1

        det = ((-2 * b * z1 - a) ** 2) - (4 * b * c)
        if det < 0:
            # The straight-line predictor from PB has no real intersection
            # with the parabolic wall. In practice this has been observed
            # to happen once PB has marched past the wall's defined z-extent
            # (z_b > x2): the parabola is only a model of the body within
            # [x1, x2], and extrapolating it further can easily stop
            # intersecting the characteristic line at all. It is not
            # necessarily a bug in this solve -- if you hit this, check
            # whether z_b below is beyond wall_params["x2"] before assuming
            # the mesh math is wrong.
            if z_b > z2 or z_b < z1:
                print(f"solve_wall_point: no real intersection (det={det:.4g}); "
                      f"z_b={z_b:.4f} is outside the wall's defined domain "
                      f"[{z1:.4f}, {z2:.4f}]")
            else:
                print(f"solve_wall_point: no real intersection (det={det:.4g}) "
                      f"at z_b={z_b:.4f}, within wall domain [{z1:.4f}, {z2:.4f}]")
            return None
        z_c_prime = (2 * b * z1 + a + np.sqrt(det)) / (2 * b)

        # Equation 2.36
        r_c_prime = r1 + ((r2 - r1) / ((z2 - z1) ** 2)) * (z_c_prime - z1) ** 2

        # Equation 2.37
        theta_c_prime = np.arctan(2 * (r2 - r1) * (z_c_prime - z1) / (z2 - z1) ** 2)

        # Equation 2.40
        q_c_a = np.sin(mu_b) * np.sin(theta_b) / (r_b * np.cos(theta_b - mu_b))
        q_c_b = theta_b - theta_c_prime
        q_c_prime = q_b * np.tan(mu_b) * (q_c_a * (z_c_prime - z_b) + q_c_b) + q_b

        # Equation 2.27
        M_c_prime = np.sqrt(2 / ((self.gamma - 1) * (((self.q_max / q_c_prime) ** 2) - 1)))

        # Equation 2.26
        mu_c_prime = np.arcsin(1 / M_c_prime)

        # ******************************************************
        # Corrector Step (iterative)
        # ******************************************************
        for _ in range(max_iters):

            # Equation 2.35b
            drdz_b = 0.5 * (np.tan(theta_b - mu_b) + np.tan(theta_c_prime - mu_c_prime))

            # Equation 2.38
            a = drdz_b
            b = (r2 - r1) / (z2 - z1) ** 2
            c = b * z1 * z1 + a * z_b - r_b + r1
            det = ((-2 * b * z1 - a) ** 2) - (4 * b * c)
            if det < 0:
                # Same failure mode as the predictor step above, but here
                # it's the *corrected* slope pushing the intersection out
                # of the wall's real-solution domain. Fail the same way
                # rather than letting sqrt() raise on a negative value.
                print(f"solve_wall_point: no real intersection (det={det:.4g}) "
                      f"during corrector iteration")
                return None
            z_c_new = (2 * b * z1 + a + np.sqrt(det)) / (2 * b)

            # Equation 2.36
            r_c_new = r1 + ((r2 - r1) / ((z2 - z1) ** 2)) * (z_c_new - z1) ** 2

            # Equation 2.37
            theta_c_new = np.arctan(2 * (r2 - r1) * (z_c_new - z1) / (z2 - z1) ** 2)

            C3 = 0.5 * ((1 / (np.tan(mu_c_prime) * q_c_prime)) + (1 / (np.tan(mu_b) * q_b)))
            C4_a = np.sin(mu_b) * np.sin(theta_b) / (r_b * np.cos(theta_b - mu_b))
            C4_b = np.sin(mu_c_prime) * np.sin(theta_c_prime) / (r_c_prime * np.cos(theta_c_prime - mu_c_prime))
            C4 = 0.5 * (C4_a + C4_b) * (z_c_new - z_b)

            # Equation 2.41
            q_c_new = ((theta_b - theta_c_new + C4) / C3) + q_b

            # Equation 2.27
            M_c_new = np.sqrt(2 / ((self.gamma - 1) * (((self.q_max / q_c_new) ** 2) - 1)))

            # Equation 2.26
            mu_c_new = np.arcsin(1 / M_c_new)

            if abs(q_c_new - q_c_prime) / q_c_prime < tol:

                break

            z_c_prime, r_c_prime, theta_c_prime, M_c_prime, mu_c_prime, q_c_prime = \
                z_c_new, r_c_new, theta_c_new, M_c_new, mu_c_new, q_c_new

        return Point(z_c_prime, r_c_prime, theta_c_prime, M_c_prime, q_c_prime)

# ---- TESTING THE CLASS ---- #
if __name__ == "__main__":

    # Define wall parameters
    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": -3.5507, "r2": -2.5}
