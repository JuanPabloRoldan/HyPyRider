import unittest
from HyPyRider.ConicalFlowAnalyzer.oblique_shock_solver import ObliqueShockSolver
import numpy as np

class TestObliqueShockSolver(unittest.TestCase):
    def test_calculate_post_shock_conditions(self):
        solver = ObliqueShockSolver(gamma=1.4)
        M1 = 2.0
        theta_s = np.radians(20)
        results = solver.calculate_post_shock_conditions(M1, theta_s)
        print(results)

if __name__ == "__main__":
    unittest.main()