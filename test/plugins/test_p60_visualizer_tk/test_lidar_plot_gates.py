"""Tests for LiDAR plot overlay gating (viz flags vs perception c41)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from avlite.plugins.p60_visualizer_tk.p69_plot_lib import LocalPlot


@pytest.fixture
def local_plot():
    plot = LocalPlot()
    yield plot
    import matplotlib.pyplot as plt

    plt.close(plot.fig)


def test_update_lidar_plot_shows_points_when_viz_enabled(local_plot):
    lidar = np.array([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    local_plot.update_lidar_plot(lidar, show_plot=True, show_global=True)
    offsets = local_plot.lidar_scatter_ax1.get_offsets()
    assert len(offsets) == 2
    np.testing.assert_allclose(offsets[0], [1.0, 2.0])


def test_update_lidar_plot_clears_when_disabled(local_plot):
    lidar = np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    local_plot.update_lidar_plot(lidar, show_plot=True, show_global=True)
    local_plot.update_lidar_plot(lidar, show_plot=False, show_global=True)
    offsets = local_plot.lidar_scatter_ax1.get_offsets()
    assert len(offsets) == 0
