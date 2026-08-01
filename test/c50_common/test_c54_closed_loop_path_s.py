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


def test_track_end_s_matches_final_arc_length_including_short_paths():
    """track_end_s is path_s[-1]; safe on 0/1-point paths (old path_s[-2] IndexError)."""
    empty = TrajectoryTracker(path=[], velocity=[])
    assert empty.track_end_s == 0.0

    one = TrajectoryTracker(path=[(0.0, 0.0)], velocity=[0.0])
    assert one.track_end_s == 0.0

    closed = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]
    tj = TrajectoryTracker(path=closed, velocity=[1.0] * len(closed))
    assert abs(tj.track_end_s - tj.path_s[-1]) < 1e-9
    # Stale [-2] workaround is one segment short of the true lap length.
    assert tj.path_s[-2] < tj.track_end_s - 1.0


def test_frenet_on_closed_loop_last_segment_keeps_s_near_track_end():
    """first==last: KD nearest at finish is index 0 — must still score the wrap segment."""
    path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    tj = TrajectoryTracker(path, velocity=[5.0] * len(path))

    # Mid last segment (0,10)→(0,0): previously snapped to s=0, d=5.
    s, d = tj.convert_xy_to_sd(0.0, 5.0)
    assert abs(d) < 1e-6
    assert abs(s - 35.0) < 1e-6

    s_np, d_np = tj.convert_xy_path_to_sd_path_np([(0.0, 5.0)])[0]
    assert abs(d_np) < 1e-6
    assert abs(s_np - 35.0) < 1e-6

    # Near finish on the wrap segment.
    s2, d2 = tj.convert_xy_to_sd(0.0, 0.5)
    assert abs(d2) < 1e-6
    assert abs(s2 - 39.5) < 1e-6


def test_bundled_yas_marina_frenet_near_finish_stays_on_wrap_segment():
    """Shipped Yas Marina race line: last ~0.3 m must not jump to s≈0 / full-lap CTE."""
    path_json = Path(__file__).resolve().parents[2] / (
        "avlite/data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    )
    data = json.loads(path_json.read_text())
    path = [tuple(pt[:2]) for pt in data["ReferenceLine"]]
    tj = TrajectoryTracker(path, velocity=list(data["ReferenceSpeed"]))

    x0, y0 = path[-2]
    x1, y1 = path[-1]
    t = 0.9
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    true_s = tj.path_s[-2] + t * (tj.path_s[-1] - tj.path_s[-2])

    s, d = tj.convert_xy_to_sd(x, y)
    assert abs(d) < 1e-3
    assert abs(s - true_s) < 1e-2
    assert s > tj.track_end_s * 0.99
