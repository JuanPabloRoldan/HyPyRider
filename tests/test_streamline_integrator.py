import unittest
from HyPyRider.CompressionSurfaceDesign.inputs.streamline_integrator import StreamlineIntegrator

class TestStreamlineIntegrator(unittest.TestCase):
    def test_streamline_integration(self):
        integrator = StreamlineIntegrator(gamma=1.405, M1=10.0, theta_s=20.0)
        print(integrator)
        integrator.tabulate_flowfield()

if __name__ == "__main__":
    unittest.main()