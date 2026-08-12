from pathlib import Path

import numpy as np

from galbot_motion_obstacle_annotator.pcd import load_pcd


def test_load_ascii_pcd():
    points = load_pcd(Path(__file__).parent / "data" / "sample_ascii.pcd")

    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[-1], [1.0, 2.0, 3.0])
