"""Headless smoke tests for the Qt visualizer (QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from avlite.c60_apps.c61_app_strategy import bootstrap_apps
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.c60_apps.c65_setting_utils import load_setting
from avlite.c60_apps.c69_settings import AppSettings


@pytest.fixture(scope="module")
def qapp():
    bootstrap_apps()
    from avlite.plugins.avlite_app_visualizer_qt.qt_binding import configure_qt_env, import_qt

    configure_qt_env(offscreen=True)
    _QtCore, _QtGui, QtWidgets = import_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    from _plugin_util import PLUGIN_DIR, PLUGIN_NAME, ensure_qt_plugin_loaded

    load_setting(AppSettings, profile="default")
    AppSettings.c62_load_plugins = True
    # Keep the in-tree community package registered across stack sync.
    AppSettings.c62_community_plugins = {PLUGIN_NAME: str(PLUGIN_DIR)}
    if "p60_visualizer_qt" not in AppSettings.c62_default_plugins:
        AppSettings.c62_default_plugins = list(AppSettings.c62_default_plugins) + ["p60_visualizer_qt"]
    load_stack_settings(profile="default", load_plugins=True)
    ensure_qt_plugin_loaded()

    from avlite.plugins.avlite_app_visualizer_qt.main_window import VisualizerQtWindow

    win = VisualizerQtWindow()
    yield win
    win.stack_panels.execution.stop_exec()
    win.log_panel.shutdown()
    win._ui_timer.stop()
    win.win.close()
    ensure_qt_plugin_loaded()


@pytest.mark.requires_data
def test_window_builds_against_basicsim(window):
    assert window.exec is not None
    assert type(window.exec.world).__name__ == "BasicSim"
    assert window.win.windowTitle().startswith("AVLite Qt")


@pytest.mark.requires_data
def test_step_and_redraw(window, qapp):
    window.stack_panels.execution.step_exec()
    window.update_ui()
    qapp.processEvents()
    ego = window.exec.ego_state
    assert isinstance(ego.x, float)
    assert float(window.setting.plot_latency_ms.get()) >= 0.0


@pytest.mark.requires_data
def test_runtime_module_switch_controller(window, qapp):
    window.setting.controller_type.set("StanleyController")
    window.reload_stack(reload_code=False)
    qapp.processEvents()
    assert window.exec is not None
    assert type(window.exec.controller).__name__ == "StanleyController"


@pytest.mark.requires_data
def test_teleport_and_spawn(window, qapp):
    from avlite.c10_perception.c11_perception_model import AgentState

    window.teleport_ego(10.0, 20.0, 0.5)
    assert abs(window.exec.ego_state.x - 10.0) < 1e-6
    agent = AgentState()
    agent.x, agent.y, agent.theta = 15.0, 20.0, 0.0
    window.spawn_agent(agent)
    window.update_ui()
    qapp.processEvents()


@pytest.mark.requires_data
def test_shortcut_mode_toggles_panels(window):
    window.setting.p60_shortcut_mode.set(True)
    window.update_shortcut_mode()
    # Offscreen windows may not be "visible" on screen; use isHidden() instead.
    assert not window.shortcut_help.isHidden()
    assert window.stack_panels.isHidden()
    window.setting.p60_shortcut_mode.set(False)
    window.update_shortcut_mode()
    assert not window.stack_panels.isHidden()
