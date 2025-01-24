import unittest
from HyPyRider.CompressionSurfaceDesign.inputs.process_LE_points import extract_points_from_file
import os

class TestProcessLEPoints(unittest.TestCase):
    def test_extract_points(self):
        file_path = "tests/test_data/LeadingEdgeData_LeftSide.nmb"
        if os.path.exists(file_path):
            df = extract_points_from_file(file_path)
            print(df)

if __name__ == "__main__":
    unittest.main()