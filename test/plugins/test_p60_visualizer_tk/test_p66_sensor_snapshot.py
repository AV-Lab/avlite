"""Exercise the view's sensor selection without creating a Tk window."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame
from avlite.plugins.p60_visualizer_tk.p66_plot_views import LocalPlanPlotView


@pytest.mark.parametrize("case", ["primary", "no_primary", "no_reading", "disabled"])
def test_plot_reads_selected_lidar_snapshot(case):
    ExecutionSettings.c41_world_capabilities = [] if case == "disabled" else None
    mount = np.eye(4)
    mount[0, 3] = 2.0
    lidar = Lidar(
        base_to_sensor=mount,
        points=None if case == "no_reading" else np.array([[3, 0, 0, 1]], dtype=np.float32),
    )
    frame = SensorFrame(
        lidars={"unused": Lidar(), "front": lidar},
        primary_lidar_name=None if case == "no_primary" else "front",
    )
    world = MagicMock()
    world.get_sensor_frame.return_value = frame
    world.get_ego_state.return_value = EgoState(x=10.0, y=20.0, theta=np.pi / 2)
    settings = MagicMock()
    settings.exec_running = False
    settings.p66_show_lidar_global.get.return_value = True
    settings.p67_show_local_global_view.get.return_value = True
    settings.p67_show_local_frenet_view.get.return_value = True
    canvas = MagicMock()
    widget = canvas.get_tk_widget.return_value
    widget.winfo_width.return_value = widget.master.winfo_width.return_value = 800
    widget.winfo_height.return_value = widget.master.winfo_height.return_value = 600
    view = SimpleNamespace(
        root=SimpleNamespace(exec=SimpleNamespace(world=world), setting=settings),
        canvas=canvas, local_plot=MagicMock(),
    )
    LocalPlanPlotView.plot(view)
    shown = view.local_plot.plot.call_args.kwargs["lidar_data"]
    if case == "primary":
        np.testing.assert_allclose(shown, [[10, 25, 0, 1]], atol=1e-6)
    else:
        assert shown is None
    assert world.get_sensor_frame.call_count == (0 if case == "disabled" else 1)
    world.get_lidar_data.assert_not_called()
    world.get_lidar_sensor.assert_not_called()
