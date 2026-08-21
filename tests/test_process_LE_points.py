import numpy as np
from src.process_LE_points import extract_points_from_file


def test_extract_points_from_file(tmp_path):
    """
    extract_points_from_file() skips a fixed 6-line header, reads a point
    count, then parses exactly that many whitespace-delimited x/y/z rows
    (ignoring any extra columns or trailing rows beyond the stated count).
    """
    file_path = tmp_path / "leading_edge.nmb"
    lines = [
        "header line 1\n",
        "header line 2\n",
        "header line 3\n",
        "header line 4\n",
        "header line 5\n",
        "header line 6\n",
        "3\n",  # number of points
        "1.0 2.0 3.0 extra_col\n",
        "4.0 5.0 6.0\n",
        "7.0 8.0 9.0\n",
        "10.0 11.0 12.0\n",  # beyond the stated count; must be ignored
    ]
    file_path.write_text("".join(lines))

    df = extract_points_from_file(str(file_path))

    assert list(df.columns) == ["X", "Y", "Z"]
    assert len(df) == 3
    assert np.allclose(df["X"], [1.0, 4.0, 7.0])
    assert np.allclose(df["Y"], [2.0, 5.0, 8.0])
    assert np.allclose(df["Z"], [3.0, 6.0, 9.0])
