"""Simple FPS / plot-latency comparison: Qt pyqtgraph vs Matplotlib Agg.

Marked ``slow`` + ``requires_data``. Run explicitly::

    QT_QPA_PLATFORM=offscreen python3 -m pytest avlite-app-visualizer-qt/test/test_fps_benchmark.py -m '' -q
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from avlite.c60_apps.c62_factory import executor_factory, load_stack_settings
from avlite.c60_apps.c65_setting_utils import load_setting
from avlite.c60_apps.c69_settings import AppSettings


@pytest.mark.slow
@pytest.mark.requires_data
def test_qt_plot_latency_vs_matplotlib_baseline(tmp_path):
    from _plugin_util import PLUGIN_DIR, PLUGIN_NAME, ensure_qt_plugin_loaded

    ensure_qt_plugin_loaded()
    load_setting(AppSettings, profile="default")
    AppSettings.c62_load_plugins = True
    AppSettings.c62_community_plugins = {PLUGIN_NAME: str(PLUGIN_DIR)}
    load_stack_settings(profile="default", load_plugins=True)
    ensure_qt_plugin_loaded()

    exec_ = executor_factory(
        executer_type="SyncExecuter",
        bridge="BasicSim",
        load_plugins=False,
    )
    # Warm one step so plans/maps exist for both backends.
    exec_.step(
        control_dt=0.01, replan_dt=0.01, perception_dt=0.01, sim_dt=0.01,
        call_replan=True, call_control=True, call_perceive=True, call_tasks=False,
        pace_perception=False, pace_replan=False, pace_control=False, pace_sim=False,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Heavier Matplotlib baseline closer to the Tk visualizer: two axes, ego box,
    # boundaries, local + global plan polylines, full canvas.draw() each frame.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    (g_plan,) = ax1.plot([], [], "b-", lw=2)
    (g_left,) = ax1.plot([], [], "orange", lw=2)
    (g_right,) = ax1.plot([], [], "tan", lw=2)
    ego_poly = plt.Polygon(np.zeros((4, 2)), fc="azure", ec="r")
    ax1.add_patch(ego_poly)
    (l_plan,) = ax2.plot([], [], "r-", lw=3)
    (l_ref,) = ax2.plot([], [], "gray", lw=1)
    n_frames = 30

    def _fill_mpl():
        plan = exec_.local_planner.global_plan if exec_.local_planner else None
        if plan is not None and plan.trajectory is not None:
            g_plan.set_data(plan.trajectory.path_x, plan.trajectory.path_y)
            g_left.set_data(plan.left_boundary_x, plan.left_boundary_y)
            g_right.set_data(plan.right_boundary_x, plan.right_boundary_y)
            l_ref.set_data(plan.trajectory.path_x, plan.trajectory.path_y)
            local = getattr(exec_.local_planner, "local_plan", None)
            traj = getattr(local, "trajectory", None) if local is not None else None
            if traj is not None:
                l_plan.set_data(traj.path_x, traj.path_y)
        ego_poly.set_xy(exec_.ego_state.get_bb_corners())

    mpl_times = []
    for _ in range(n_frames):
        exec_.step(
            control_dt=0.01, replan_dt=0.01, perception_dt=0.01, sim_dt=0.01,
            call_replan=True, call_control=True, call_perceive=True, call_tasks=False,
            pace_perception=False, pace_replan=False, pace_control=False, pace_sim=False,
        )
        t0 = time.perf_counter()
        _fill_mpl()
        fig.canvas.draw()
        mpl_times.append((time.perf_counter() - t0) * 1000.0)
    plt.close(fig)
    mpl_latency = sum(mpl_times) / len(mpl_times)
    mpl_fps = 1000.0 / max(mpl_latency, 1e-6)

    from avlite.plugins.avlite_app_visualizer_qt.qt_binding import configure_qt_env, import_qt
    from avlite.plugins.avlite_app_visualizer_qt.main_window import VisualizerQtWindow

    configure_qt_env(offscreen=True)
    _QtCore, _QtGui, QtWidgets = import_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = VisualizerQtWindow()
    win.exec = exec_

    qt_times = []
    for _ in range(n_frames):
        exec_.step(
            control_dt=0.01, replan_dt=0.01, perception_dt=0.01, sim_dt=0.01,
            call_replan=True, call_control=True, call_perceive=True, call_tasks=False,
            pace_perception=False, pace_replan=False, pace_control=False, pace_sim=False,
        )
        t0 = time.perf_counter()
        latency = win.global_plot.redraw() + win.local_plot.redraw()
        app.processEvents()
        qt_times.append(latency if latency > 0 else (time.perf_counter() - t0) * 1000.0)
    qt_latency = sum(qt_times) / len(qt_times)
    qt_fps = 1000.0 / max(qt_latency, 1e-6)

    report = (
        f"Matplotlib Agg plot latency≈{mpl_latency:.2f} ms ({mpl_fps:.1f} FPS equiv) | "
        f"Qt pyqtgraph plot latency≈{qt_latency:.2f} ms ({qt_fps:.1f} FPS equiv)"
    )
    (tmp_path / "fps_report.txt").write_text(report + "\n")
    print("\n" + report)

    win.stack_panels.execution.stop_exec()
    win.log_panel.shutdown()
    win._ui_timer.stop()
    win.win.close()
    ensure_qt_plugin_loaded()

    # Offscreen software path still should stay interactive; require Qt latency
    # not worse than the heavier Matplotlib baseline by more than 10%.
    assert qt_latency <= mpl_latency * 1.1, report
    assert qt_latency < 33.0, report  # able to sustain ≥30 Hz plot refresh
