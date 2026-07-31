"""Closed-loop reference paths must keep monotonic arc-length path_s."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def test_exact_duplicate_endpoints_keep_monotonic_path_s():
    """first==last must not snap path_s via KD-tree Frenet reproject of the reference."""
    path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    tj = TrajectoryTracker(path, velocity=[5.0] * len(path))
    ps = np.asarray(tj.path_s, dtype=float)

    assert float(ps[0]) == 0.0
    assert abs(float(ps[-1]) - 40.0) < 1e-6
    assert float(ps[-1]) == float(np.max(ps))
    assert bool(np.all(np.diff(ps) >= -1e-12))


def test_bundled_yas_marina_race_line_path_s_monotonic():
    """Shipped closed race line previously initialized with path_s[-1] == 0."""
    path_json = Path(__file__).resolve().parents[2] / (
        "avlite/data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    )
    data = json.loads(path_json.read_text())
    path = [tuple(pt[:2]) for pt in data["ReferenceLine"]]
    assert path[0] == path[-1]
    tj = TrajectoryTracker(path, velocity=list(data["ReferenceSpeed"]))
    ps = np.asarray(tj.path_s, dtype=float)
    assert float(ps[0]) == 0.0
    assert float(ps[-1]) > 1.0
    assert float(ps[-1]) == float(np.max(ps))
    assert bool(np.all(np.diff(ps) >= -1e-12))


def test_frenet_after_corner_keeps_on_path_cte_near_zero():
    """On-path points after a 90° turn must not inherit the previous segment's CTE."""
    path = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
    tj = TrajectoryTracker(path, velocity=[5.0] * len(path))
    s, d = tj.convert_xy_to_sd(50.0, 25.0)
    assert abs(d) < 1e-6
    assert abs(s - 75.0) < 1e-6
    s2, d2 = tj.convert_xy_path_to_sd_path_np([(50.0, 25.0)])[0]
    assert abs(d2) < 1e-6
    assert abs(s2 - 75.0) < 1e-6


def test_convert_sd_to_xy_brackets_by_arc_length():
    path = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
    tj = TrajectoryTracker(path, velocity=[5.0] * len(path))
    x, y = tj.convert_sd_to_xy(75.0, 0.0)
    assert abs(x - 50.0) < 1e-6
    assert abs(y - 25.0) < 1e-6
