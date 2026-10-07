"""Verify predicted trajectories are plotted as full polylines, not 2-point segments."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    EgoState,
    GMM,
    GP,
    MultiTrajectory,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.plugins.p60_visualizer_tk.p69_plot_lib import LocalPlot
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


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


def _segment_xy(plot: LocalPlot, index: int = 0):
    seg = plot.prediction_lines_ax1.get_segments()[index]
    return seg[:, 0], seg[:, 1]


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

        xdata, ydata = _segment_xy(plot)
        assert len(xdata) == n_steps + 1
        assert len(ydata) == n_steps + 1

        # Midpoint of plotted path should deviate from straight chord (curved LSTM-like output).
        mid_idx = len(xdata) // 2
        chord_y = (ydata[0] + ydata[-1]) / 2.0
        assert abs(ydata[mid_idx] - chord_y) > 0.5

        plt.close(plot.fig)

    def test_behind_agent_prediction_uses_distinct_color(self):
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

        xdata, ydata = _segment_xy(plot)
        # Behind-ego predictions are now drawn (agent pose + 3 predicted steps)…
        assert len(xdata) == len(trajectories[0]) + 1
        assert len(ydata) == len(trajectories[0]) + 1
        # …in the distinct "behind" colour rather than the ahead colour.
        rgba = plot.prediction_lines_ax1.get_edgecolors()[0]
        assert np.allclose(rgba, to_rgba(LocalPlot.PREDICTION_BEHIND_COLOR))

        plt.close(plot.fig)

    def test_ahead_agent_prediction_uses_ahead_color(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        trajectories = _curved_trajectory((10.0, 0.0), radius=8.0, n_steps=6)[None, :, :]

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

        rgba = plot.prediction_lines_ax1.get_edgecolors()[0]
        assert np.allclose(rgba, to_rgba(LocalPlot.PREDICTION_AHEAD_COLOR))

        plt.close(plot.fig)

    def test_lower_mode_weight_is_more_transparent(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        modes = np.array([
            [[11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
            [[11.0, 1.0], [12.0, 1.0], [13.0, 1.0]],
        ])
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=MultiTrajectory(
                trajectories={0: modes},
                weights={0: np.array([0.8, 0.2])},
            ),
        )

        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )

        rgba = plot.prediction_lines_ax1.get_edgecolors()
        assert len(plot.prediction_lines_ax1.get_segments()) == 2
        assert abs(rgba[0, 3] - 0.8) < 1e-6
        assert abs(rgba[1, 3] - 0.2) < 1e-6
        assert np.allclose(rgba[0, :3], to_rgba(LocalPlot.PREDICTION_AHEAD_COLOR)[:3])

        plt.close(plot.fig)

    def test_gp_mean_is_one_opaque_line(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        mean = np.array([[11.0, 0.0], [12.0, 0.0], [13.0, 0.0]])
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GP(means={0: mean}, covariance={0: np.eye(6)}),
        )

        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )

        assert len(plot.prediction_lines_ax1.get_segments()) == 1
        xdata, _ydata = _segment_xy(plot)
        assert len(xdata) == len(mean) + 1
        rgba = plot.prediction_lines_ax1.get_edgecolors()[0]
        assert abs(rgba[3] - 1.0) < 1e-6

        # eye(6) is isotropic, so each contour is a circle on the last mean.
        # Only the last step is inside the 0.5 s stride when dt is 0.1 s.
        paths = plot.prediction_regions_ax1.get_paths()
        assert len(paths) == 2
        center = mean[-1]
        radii = sorted(
            float(np.max(np.hypot(path.vertices[:, 0] - center[0], path.vertices[:, 1] - center[1])))
            for path in paths
        )
        expected = [
            np.sqrt(-2.0 * np.log(1.0 - p))
            for p in LocalPlot.PREDICTION_PERCENTILES
        ]
        np.testing.assert_allclose(radii, expected, atol=1e-6)
        edge_alpha = sorted(plot.prediction_regions_ax1.get_edgecolors()[:, 3])
        np.testing.assert_allclose(edge_alpha, sorted(LocalPlot.PREDICTION_PERCENTILES), atol=1e-6)

        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=False,
        )
        assert plot.prediction_regions_ax1.get_paths() == []

        plt.close(plot.fig)

    def test_gp_ellipse_is_longer_along_the_heading(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        mean = np.array([[12.0, 0.0]])
        cov = np.diag([4.0, 1.0])
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GP(means={0: mean}, covariance={0: cov}, predict_delta_t=0.5),
        )
        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )
        paths = plot.prediction_regions_ax1.get_paths()
        assert len(paths) == 2
        for path in paths:
            offset = path.vertices - mean[0]
            assert np.max(np.abs(offset[:, 0])) > np.max(np.abs(offset[:, 1])) * 1.5
        plt.close(plot.fig)

    def test_gmm_ellipses_follow_mode_weight_and_percentile(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=0)
        modes = np.array([
            [[11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
            [[11.0, 1.0], [12.0, 1.0], [13.0, 1.0]],
        ])
        weights = np.array([0.8, 0.2])
        cov = np.zeros((2, 3, 2, 2))
        cov[:, :, 0, 0] = 1.0
        cov[:, :, 1, 1] = 1.0
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GMM(
                trajectories={0: modes},
                weights={0: weights},
                covariances={0: cov},
                predict_delta_t=0.1,
            ),
        )
        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.update_perception_model_plots(
            exec_pm=pm,
            global_trajectory=_straight_reference_path(),
            show_plot=True,
            show_prediction=True,
        )
        line_alpha = plot.prediction_lines_ax1.get_edgecolors()[:, 3]
        np.testing.assert_allclose(line_alpha, weights)
        edge_alpha = plot.prediction_regions_ax1.get_edgecolors()[:, 3]
        expected = [w * p for w in weights for p in LocalPlot.PREDICTION_PERCENTILES]
        np.testing.assert_allclose(edge_alpha, expected)
        assert edge_alpha[2] < edge_alpha[0]
        assert edge_alpha[3] < edge_alpha[1]
        plt.close(plot.fig)

    def test_occupancy_flow_alpha_is_the_cell_probability(self):
        low = np.zeros((1, 2))
        low[0, 0] = 0.2
        low[0, 1] = 0.8
        high = np.zeros((1, 2))
        high[0, 1] = 0.4
        pm = PerceptionModel(
            prediction=OccupancyFlow(
                occupancy_flow={0: [low, high]},
                origin_x=1.0,
                origin_y=2.0,
                resolution=0.5,
            ),
        )
        plot = LocalPlot(max_plan_length=1, max_agent_count=1)
        plot.ax1.set_xlim(-30.0, 30.0)
        plot.ax1.set_ylim(-10.0, 10.0)
        plot.update_pm_occupancy_flow_plots(pm, show_plot=True)
        np.testing.assert_allclose(plot.ax1.get_xlim(), (-30.0, 30.0))
        np.testing.assert_allclose(plot.ax1.get_ylim(), (-10.0, 10.0))
        assert plot.ax1.get_aspect() == 1.0
        face_alpha = plot.occupancy_flow_cells_ax1.get_facecolors()[:, 3]
        np.testing.assert_allclose(face_alpha, [0.2, 0.8])
        assert face_alpha[0] < face_alpha[1]
        assert plot.occupancy_flow_cells_ax2.get_paths() == []
        plt.close(plot.fig)

    def test_gp_and_gmm_do_not_draw_occupancy_cells(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=1.0, agent_id=0)
        mean = np.array([[11.0, 0.0], [12.0, 0.0]])
        gp = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0),
            agent_vehicles=[agent],
            prediction=GP(
                predict_delta_t=0.5,
                means={0: mean},
                covariance={0: np.eye(4)},
            ),
        )
        gmm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0),
            agent_vehicles=[agent],
            prediction=GMM(
                predict_delta_t=0.5,
                trajectories={0: mean[None, :, :]},
                weights={0: np.array([1.0])},
                covariances={0: np.stack([np.eye(2), np.eye(2)])[None, :, :, :]},
            ),
        )
        for pm in (gp, gmm):
            plot = LocalPlot(max_plan_length=1, max_agent_count=1)
            plot.update_perception_model_plots(
                exec_pm=pm,
                global_trajectory=_straight_reference_path(),
                show_plot=True,
                show_prediction=True,
            )
            plot.update_pm_occupancy_flow_plots(
                pm, show_plot=True, global_trajectory=_straight_reference_path(), show_frenet=True,
            )
            assert len(plot.prediction_regions_ax1.get_paths()) > 0
            assert plot.occupancy_flow_cells_ax1.get_paths() == []
            assert plot.occupancy_flow_cells_ax2.get_paths() == []
            plt.close(plot.fig)
