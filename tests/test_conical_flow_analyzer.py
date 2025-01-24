import unittest
from HyPyRider.ConicalFlowAnalyzer.conical_flow_analyzer import ConicalFlowAnalyzer
import numpy as np

class TestConicalFlowAnalyzer(unittest.TestCase):

    def test_solve_taylor_maccoll(self):
        analyzer = ConicalFlowAnalyzer(gamma=1.405, M1=10.0)
        theta_s_deg = 30
        cone_angle, _, _ = analyzer.solve_taylor_maccoll(theta_s_deg)
        self.assertAlmostEqual(cone_angle, 39.6, places=1)  # Replace with actual expected value

    def test_solve_taylor_maccoll_range(self):
        analyzer = ConicalFlowAnalyzer(gamma=1.405, M1=10.0)
        theta_s_range = np.linspace(6, 73, 200)
        data = analyzer.solve_taylor_maccoll_range(theta_s_range)
        self.assertIsNotNone(data)

    def test_plot_shock_vs_cone(self):
        analyzer = ConicalFlowAnalyzer(gamma=1.405, M1=10.0)
        theta_s_range = np.linspace(6, 73, 200)
        data = analyzer.solve_taylor_maccoll_range(theta_s_range)
        analyzer.plot_shock_vs_cone(data)  # No assertions; visual test

if __name__ == "__main__":
    unittest.main()