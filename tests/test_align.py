import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.labels.align import align_points  # noqa: E402
from pathsnap.geo import point_seg_dist  # noqa: E402


def _route():
    # out-and-back loop: east along y=0, up to y=8, back west along y=8 (close to the outbound leg)
    pts = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [30, 8], [20, 8], [10, 8], [0, 8]], float)
    return pts[:-1], pts[1:]


def test_monotonic_alignment_on_loop():
    ra, rb = _route()
    out = np.column_stack([np.arange(1, 26, 2.0), np.full(13, 3.0)])    # outbound, y=3 (nearer y=0)
    back = np.column_stack([np.arange(25, 0, -2.0), np.full(13, 5.0)])  # return, y=5 (nearer y=8)
    up = np.array([[30.5, 4.0]])
    p = np.vstack([out, up, back])
    pos, d = align_points(p, ra, rb, half=1)  # tiny band forces the widening path
    assert np.all(np.diff(pos) >= 0)
    exp = np.concatenate([np.minimum(out[:, 0] // 10, 2), [3], 4 + np.minimum((30 - back[:, 0]) // 10, 2)])
    assert np.array_equal(pos, exp.astype(int)), (pos, exp)
    assert np.allclose(d[:13], 3) and np.allclose(d[14:], 3)


def test_point_seg_dist():
    d, t = point_seg_dist(np.array([[5.0, 3.0], [-4.0, 3.0]]), np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert np.allclose(d, [3, 5]) and np.allclose(t, [0.5, 0])
