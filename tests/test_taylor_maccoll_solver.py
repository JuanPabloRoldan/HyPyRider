import unittest
from HyPyRider.ConicalFlowAnalyzer.taylor_maccoll_solver import TaylorMaccollSolver

class TestTaylorMaccollSolver(unittest.TestCase):
    def test_taylor_maccoll_solve(self):
        solver = TaylorMaccollSolver(gamma=1.4)
        theta0 = 0.0
        Vr0 = 0.5
        dVr0 = -0.1
        theta_values, Vr_values, dVr_values = solver.solve(theta0, Vr0, dVr0)
        print(theta_values, Vr_values, dVr_values)

if __name__ == "__main__":
    unittest.main()