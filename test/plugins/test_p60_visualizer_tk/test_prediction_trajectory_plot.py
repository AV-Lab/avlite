"""Verify predicted trajectories are plotted as full polylines, not 2-point segments."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    EgoState,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.plugins.p50_visualizer_tk.p59_plot_lib import LocalPlot
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker


def _straight_reference_path(n: int = 50) -> TrajectoryTracker:
    xs = np.linspace(0.0, 100.0, n)
    path = [(float(x), 0.0) for x in xs]
    return TrajectoryTracker(path=path, velocity=[5.0] * n)


def _curved_trajectory(center: tuple[float, float], radius: float, n_steps: int) -> np.ndarray:
    """Semicircle arc in front of ego (positive x from agent)."""
    cx, cy = center
    angles = np.linspace(-np.pi / 4, np.pi / 4, n_steps)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.column_stack([xs, ys])


class TestPredictionTrajectoryPlot:
    def test_plots_full_trajectory_polyline_not_two_points(self):
        n_steps = 10
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        trajectories = _curved_trajectory((10.0, 0.0), radius=8.0, n_steps=n_steps)[None, :, :]

        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=SingleTrajectory(trajectories={0: trajectories[0]}),
        )

        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )

        line = plot.prediction_lines_ax1[0]
        xdata, ydata = line.get_data()
        assert len(xdata) == n_steps + 1
        assert len(ydata) == n_steps + 1

        # Midpoint of plotted path should deviate from straight chord (curved LSTM-like output).
        mid_idx = len(xdata) // 2
        chord_y = (ydata[0] + ydata[-1]) / 2.0
        assert abs(ydata[mid_idx] - chord_y) > 0.5

        plt.close(plot.fig)

    def test_hides_prediction_for_agent_behind_ego(self):
        agent = AgentState(x=-5.0, y=0.0, theta=np.pi, velocity=5.0, agent_id=0)
        trajectories = np.array([[[-4.0, 0.0], [-3.0, 0.0], [-2.0, 0.0]]], dtype=float)

        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=SingleTrajectory(trajectories={0: trajectories[0]}),
        )

        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )

        xdata, ydata = plot.prediction_lines_ax1[0].get_data()
        assert len(xdata) == 0
        assert len(ydata) == 0

        plt.close(plot.fig)
