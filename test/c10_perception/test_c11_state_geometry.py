"""Regression tests for State bounding-box corner geometry."""

import math

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import State


def test_axis_aligned_bb_corners():
    state = State(x=10.0, y=-2.0, theta=0.0, length=4.0, width=2.0)
    corners = state.get_bb_corners()
    expected = np.array(
        [
            [8.0, -3.0],
            [12.0, -3.0],
            [12.0, -1.0],
            [8.0, -1.0],
        ]
    )
    np.testing.assert_allclose(corners, expected, atol=1e-9)
    assert state.get_bb_polygon().contains(state.get_bb_polygon().centroid)
    # Explicit center containment via shapely
    from shapely.geometry import Point

    assert state.get_bb_polygon().contains(Point(state.x, state.y))


def test_yaw_90_swaps_length_and_width_axes():
    state = State(x=0.0, y=0.0, theta=math.pi / 2, length=4.0, width=2.0)
    corners = state.get_bb_corners()
    # Body (cx, cy) → world (-cy, cx) at θ=π/2.
    expected = np.array(
        [
            [1.0, -2.0],
            [1.0, 2.0],
            [-1.0, 2.0],
            [-1.0, -2.0],
        ]
    )
    np.testing.assert_allclose(corners, expected, atol=1e-9)
    xs, ys = corners[:, 0], corners[:, 1]
    assert xs.min() == pytest.approx(-1.0)
    assert xs.max() == pytest.approx(1.0)
    assert ys.min() == pytest.approx(-2.0)
    assert ys.max() == pytest.approx(2.0)
