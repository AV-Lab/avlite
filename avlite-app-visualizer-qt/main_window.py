"""Main Qt5 visualizer window — feature parity target for the Tk app."""

from __future__ import annotations

import logging
import time
from typing import Optional

from avlite import __version__
from avlite.c10_perception.c11_perception_model import AgentState
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import executor_factory, load_stack_settings
from avlite.c60_apps.c63_plugins import reload_lib
from avlite.c60_apps.c65_setting_utils import list_profiles, load_setting
from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c69_settings import AppSettings

from .log_panel import LogPanel
from .plot_widgets import GlobalPlotWidget, LocalPlotWidget
from .qt_binding import configure_qt_env, import_pyqtgraph, import_qt
from .setting_model import VisualizationSettings
from .shortcuts import SHORTCUTS
from .stack_panels import StackPanels

log = logging.getLogger(__name__)


class VisualizerQtWindow:
    """Owns the QMainWindow, stack, plots, and UI refresh timer.

    Sim / planning work is driven by ``ExecutionPanel`` calling ``exec.step``
    (AsyncThreadedExecuter keeps heavy work off the UI thread). Plot redraws
    run on a dedicated QTimer at ``p60_ui_refresh_hz`` and only call
    ``PlotDataItem.setData`` — never Matplotlib.
    """

    hosting_plugin_name = "avlite-app-visualizer-qt"

    def __init__(self):
        configure_qt_env()
        QtCore, QtGui, QtWidgets = import_qt()
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets
        import_pyqtgraph()  # OpenGL on when available; disabled offscreen via env

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(f"AVLite Qt Visualizer v{__version__}")
        self.win.resize(1400, 900)
        self.exec = None
        self.setting = VisualizationSettings()
        self.loaded_profile: Optional[str] = None
        self._ui_frames = 0
        self._ui_t0 = time.perf_counter()

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        self.plots_splitter = QtWidgets.QSplitter()
        self.local_plot = LocalPlotWidget(self)
        self.global_plot = GlobalPlotWidget(self)
        self.plots_splitter.addWidget(self.local_plot)
        self.plots_splitter.addWidget(self.global_plot)
        self.plots_splitter.setStretchFactor(0, 1)
        self.plots_splitter.setStretchFactor(1, 1)
        root_layout.addWidget(self.plots_splitter, 1)

        self.shortcut_help = QtWidgets.QLabel()
        self.shortcut_help.setWordWrap(True)
        self.shortcut_help.setVisible(False)
        self.shortcut_help.setText(
            "  |  ".join(f"{s.default}: {s.label}" for s in SHORTCUTS)
        )
        root_layout.addWidget(self.shortcut_help)

        self.stack_panels = StackPanels(self)
        root_layout.addWidget(self.stack_panels)

        self.log_panel = LogPanel(self)
        root_layout.addWidget(self.log_panel, 0)

        self._build_menu()
        self.win.closeEvent = self._on_close  # type: ignore[method-assign]
        self.win.keyPressEvent = self._on_key  # type: ignore[method-assign]

        self._ui_timer = QtCore.QTimer(self.win)
        self._ui_timer.timeout.connect(self.update_ui)
        hz = max(5.0, float(self.setting.p60_ui_refresh_hz))
        self._ui_timer.start(int(1000 / hz))

        self.setting.profile_list = list_profiles(AppSettings)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.setting.c60_selected_profile.set(startup)

        self.load_settings()
        self.reload_stack(reload_code=False)
        self._apply_theme()
        self.update_views()

    def _build_menu(self) -> None:
        QtWidgets = self.QtWidgets
        menubar = self.win.menuBar()
        file_menu = menubar.addMenu("&File")
        act_settings = file_menu.addAction("Settings / Profiles…")
        act_settings.triggered.connect(self._open_profile_dialog)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Exit")
        act_quit.triggered.connect(self.win.close)

        view_menu = menubar.addMenu("&View")
        act_global = view_menu.addAction("Toggle global plot")
        act_global.triggered.connect(
            lambda: (
                self.setting.p67_global_plan_view.set(not self.setting.p67_global_plan_view.get()),
                self.update_views(),
            )
        )
        act_local = view_menu.addAction("Toggle local plot")
        act_local.triggered.connect(
            lambda: (
                self.setting.p67_show_local_global_view.set(
                    not self.setting.p67_show_local_global_view.get()
                ),
                self.update_views(),
            )
        )
        act_dark = view_menu.addAction("Toggle dark mode")
        act_dark.triggered.connect(self._toggle_dark)

        help_menu = menubar.addMenu("&Help")
        act_about = help_menu.addAction("About AVLite")
        act_about.triggered.connect(self._show_about)

    def _show_about(self) -> None:
        self.QtWidgets.QMessageBox.about(
            self.win,
            "About AVLite",
            f"<h3>AVLite</h3><p>Version {__version__}</p>"
            "<p>Qt5 visualizer plugin (pyqtgraph).</p>",
        )

    def _toggle_dark(self) -> None:
        self.setting.p60_dark_mode.set(not self.setting.p60_dark_mode.get())
        self._apply_theme()

    def _apply_theme(self) -> None:
        dark = bool(self.setting.p60_dark_mode.get())
        if dark:
            self.setting.p60_bg_color = "#1e1e1e"
            self.setting.p60_fg_color = "#dddddd"
            self.win.setStyleSheet(
                "QWidget { background: #2b2b2b; color: #dddddd; }"
                "QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
                "QPlainTextEdit { background: #1a1a1a; color: #cfcfcf; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #3a3a3a; }"
            )
        else:
            self.setting.p60_bg_color = "#f5f5f5"
            self.setting.p60_fg_color = "#222222"
            self.win.setStyleSheet("")
        self.local_plot.update_theme(self.setting.p60_bg_color, self.setting.p60_fg_color)
        self.global_plot.update_theme(self.setting.p60_bg_color, self.setting.p60_fg_color)

    def _open_profile_dialog(self) -> None:
        profiles = list_profiles(AppSettings) or ["default"]
        current = self.setting.c60_selected_profile.get()
        name, ok = self.QtWidgets.QInputDialog.getItem(
            self.win, "Profiles", "Load profile:", profiles,
            max(0, profiles.index(current) if current in profiles else 0), False,
        )
        if ok and name:
            self.load_settings(profile=name)
            self.reload_stack(reload_code=False)

    def show(self) -> None:
        self.win.show()

    def exec_app(self) -> int:
        self.show()
        return int(self.app.exec_())

    def _on_close(self, event) -> None:
        try:
            self.stack_panels.execution.stop_exec()
            self.log_panel.shutdown()
            self._ui_timer.stop()
        finally:
            event.accept()

    def _on_key(self, event) -> None:
        key = event.key()
        text = event.text()
        mods = event.modifiers()
        Qt = self.QtCore.Qt

        def ignore_if_typing() -> bool:
            w = self.app.focusWidget()
            return isinstance(w, (self.QtWidgets.QLineEdit, self.QtWidgets.QPlainTextEdit, self.QtWidgets.QSpinBox))

        if ignore_if_typing() and key not in (Qt.Key_Escape,):
            return self.QtWidgets.QMainWindow.keyPressEvent(self.win, event)

        if key == Qt.Key_Q:
            self.win.close()
            return
        if text == "S" or (key == Qt.Key_S and mods & Qt.ShiftModifier):
            self.setting.p60_shortcut_mode.set(not self.setting.p60_shortcut_mode.get())
            self.update_shortcut_mode()
            return
        if key == Qt.Key_F:
            self.switch_profile()
            return
        if key == Qt.Key_R and not (mods & Qt.ShiftModifier):
            self.reload_stack(reload_code=True)
            return
        if key == Qt.Key_X:
            self.stack_panels.execution.toggle_exec()
            return
        if key == Qt.Key_C:
            self.stack_panels.execution.step_exec()
            return
        if key == Qt.Key_T:
            self.stack_panels.execution.reset_exec()
            return
        if key == Qt.Key_L:
            self.log_panel.clear()
            return
        if key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.setting.p66_global_zoom = max(5.0, self.setting.p66_global_zoom * 0.9)
            self.setting.p66_xy_zoom = max(5.0, self.setting.p66_xy_zoom * 0.9)
            return
        if key == Qt.Key_Minus:
            self.setting.p66_global_zoom = min(5000.0, self.setting.p66_global_zoom * 1.1)
            self.setting.p66_xy_zoom = min(500.0, self.setting.p66_xy_zoom * 1.1)
            return
        if text == "w":
            self.stack_panels.control._drive(3, 0)
            return
        if text == "s":
            self.stack_panels.control._drive(-3, 0)
            return
        if text == "a":
            self.stack_panels.control._drive(0, 0.7)
            return
        if text == "d":
            self.stack_panels.control._drive(0, -0.7)
            return
        if text == "h":
            self.stack_panels.control._step()
            return
        if text == "i":
            self.stack_panels.control._align()
            return
        if text == "r":
            self.stack_panels.planning._local_replan()
            return
        if text == "n" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(1)
            self.update_ui()
            return
        if text == "b" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(-1)
            self.update_ui()
            return
        return self.QtWidgets.QMainWindow.keyPressEvent(self.win, event)

    def update_shortcut_mode(self) -> None:
        short = bool(self.setting.p60_shortcut_mode.get())
        self.shortcut_help.setVisible(short)
        self.stack_panels.setVisible(not short)

    def update_views(self) -> None:
        show_g = bool(self.setting.p67_global_plan_view.get())
        show_l = bool(self.setting.local_plan_view_enabled())
        self.global_plot.setVisible(show_g)
        self.local_plot.setVisible(show_l)
        self.local_plot.update_visibility()
        self.update_shortcut_mode()

    def local_plan_view_enabled(self) -> bool:
        return self.setting.local_plan_view_enabled()

    def update_ui(self) -> None:
        if self.exec is None:
            return
        t0 = time.perf_counter()
        latency = 0.0
        if self.setting.p67_global_plan_view.get():
            latency += self.global_plot.redraw()
        if self.setting.local_plan_view_enabled():
            latency += self.local_plot.redraw()

        ego = self.exec.ego_state
        self.setting.vehicle_state.set(
            f"Loc: ({ego.x:+7.2f}, {ego.y:+7.2f}), Vel: {ego.velocity:5.2f} "
            f"({ego.velocity * 3.6:6.2f} km/h), θ: {ego.theta:+5.1f}"
        )
        if self.exec.local_planner is not None:
            self.setting.current_wp.set(str(self.exec.local_planner.global_trajectory.current_wp))
            self.setting.lap.set(f"{self.exec.local_planner.lap:5d}")
        if self.exec.controller is not None:
            self.setting.cte_vel.set(float(getattr(self.exec.controller, "cte_velocity", 0.0) or 0.0))
            self.setting.cte_steer.set(float(getattr(self.exec.controller, "cte_steer", 0.0) or 0.0))
            cmd = getattr(self.exec.controller, "cmd", None)
            if cmd is not None:
                self.setting.cmd_accel.set(float(getattr(cmd, "acceleration", 0.0) or 0.0))
                self.setting.cmd_steer.set(float(getattr(cmd, "steer", 0.0) or 0.0))
        self.setting.elapsed_real_time.set(f"{self.exec.elapsed_real_time:6.2f}")
        self.setting.elapsed_sim_time.set(f"{self.exec.elapsed_sim_time:6.2f}")
        self.setting.replan_fps.set(f"{self.exec.planner_fps:6.1f}")
        self.setting.control_fps.set(f"{self.exec.control_fps:6.1f}")
        self.setting.perception_fps.set(f"{self.exec.perception_fps:6.1f}")
        self.setting.plot_latency_ms.set(f"{latency:5.1f}")

        self._ui_frames += 1
        elapsed = time.perf_counter() - self._ui_t0
        if elapsed >= 0.5:
            self.setting.ui_fps.set(f"{self._ui_frames / elapsed:5.1f}")
            self._ui_frames = 0
            self._ui_t0 = time.perf_counter()
        log.debug("UI update %.2f ms (plot %.2f)", (time.perf_counter() - t0) * 1000, latency)

    def apply_global_plan(
        self, global_plan: GlobalPlan, ego_xy: Optional[tuple[float, float]] = None
    ) -> None:
        if self.exec is None:
            return
        if "apply_global_plan" in vars(type(self.exec)):
            self.exec.apply_global_plan(global_plan, ego_xy=ego_xy)
            return
        if global_plan is None or global_plan.trajectory is None or len(global_plan.trajectory.path_s) == 0:
            log.error("apply_global_plan: invalid plan")
            return
        ego_xy = ego_xy if ego_xy is not None else (self.exec.ego_state.x, self.exec.ego_state.y)
        if self.exec.local_planner:
            self.exec.local_planner.set_global_plan(global_plan, ego_xy=ego_xy)
        if self.exec.controller:
            self.exec.controller.set_trajectory_tracker(global_plan.trajectory)
            self.exec.controller.reset()

    def replan_global(self) -> None:
        if self.exec is None or not self.exec.global_planner:
            log.error("Global replan failed: no global planner.")
            return
        goal = self.exec.global_planner.goal_point
        if goal is not None:
            self.exec.global_planner.set_start_goal((self.exec.ego_state.x, self.exec.ego_state.y), goal)
        new_plan = self.exec.global_planner.plan(
            perception_model=self.exec.pm,
            sensors=self.exec.world.get_sensor_frame(),
        )
        if new_plan is None or new_plan.trajectory is None:
            log.error("Global replan failed: planner returned no valid plan.")
            return
        self.apply_global_plan(new_plan)
        self.update_ui()

    def teleport_ego(self, x, y, theta=None) -> None:
        self.exec.world.teleport_ego(x, y, theta)
        self.exec.pm.ego_vehicle.copy_from(self.exec.world.get_ego_state())

    def apply_world_control(self, cmd, dt) -> None:
        self.exec.world.control_ego_state(cmd=cmd, dt=dt)
        self.exec.pm.ego_vehicle.copy_from(self.exec.world.get_ego_state())

    def spawn_agent(self, agent_state: AgentState) -> None:
        if self.exec is None:
            return
        global_plan = self.exec.local_planner.global_plan if self.exec.local_planner else None
        self.exec.world.spawn_agent(agent_state, global_plan=global_plan)

    def _pin_self_as_community_plugin(self) -> None:
        """Keep this package listed so stack sync does not unload the running UI."""
        from pathlib import Path

        plugin_dir = str(Path(__file__).resolve().parent)
        plugins = dict(AppSettings.c62_community_plugins or {})
        plugins.setdefault(self.hosting_plugin_name, plugin_dir)
        AppSettings.c62_community_plugins = plugins
        defaults = list(AppSettings.c62_default_plugins or [])
        if "p60_visualizer_qt" not in defaults:
            defaults.append("p60_visualizer_qt")
            AppSettings.c62_default_plugins = defaults

    def load_settings(self, only_stack: bool = False, profile: str | None = None) -> None:
        if profile:
            self.setting.c60_selected_profile.set(profile)
        else:
            profile = self.setting.c60_selected_profile.get()
        load_setting(AppSettings, profile=profile)
        self._pin_self_as_community_plugin()
        self.setting.sync_app_from_singleton()
        if not only_stack:
            load_setting(self.setting.schema, profile=profile)
            # Push plugin schema defaults into runtime vars where applicable
            from .settings import PluginSettings

            self.setting.p60_dark_mode.set(bool(PluginSettings.p60_dark_mode))
            self.setting.p60_ui_refresh_hz = float(PluginSettings.p60_ui_refresh_hz)
            self.setting.p66_xy_zoom = float(PluginSettings.p66_xy_zoom)
            self.setting.p66_frenet_zoom = float(PluginSettings.p66_frenet_zoom)
            self.setting.p66_global_zoom = float(PluginSettings.p66_global_zoom)
        load_stack_settings(profile=profile)
        self._pin_self_as_community_plugin()
        self.setting.sync_stack_from_singletons()
        ConfigPaths.set_startup_profile(profile)
        self.loaded_profile = profile
        self.log_panel.clear()
        log.info("Loaded settings from profile: %s", profile)

    def refresh_pipeline(self) -> None:
        if self.exec is None:
            return
        from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline as PercPipe
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningPipeline as LocPipe

        if type(self.exec.perception).__name__ == "PerceptionPipeline":
            self.exec.perception = PercPipe(self.exec.pm)
        if self.exec.local_planner is not None and type(self.exec.local_planner).__name__ == "LocalPlanningPipeline":
            lp = self.exec.local_planner
            self.exec.local_planner = LocPipe(global_plan=lp.global_plan, env=self.exec.pm)
        try:
            self.exec.validate_stack()
        except ValueError as e:
            log.error("Pipeline refresh failed: %s", e)
        self.update_ui()

    def reload_stack(self, reload_code: bool = True) -> None:
        self.stack_panels.execution.stop_exec()
        error = None
        try:
            if reload_code:
                reload_lib(exclude_settings=True, reload_plugins=self.setting.c62_load_plugins.get())
            self.exec = executor_factory(
                executer_type=self.setting.executer_type.get(),
                bridge=self.setting.execution_bridge.get(),
                perception_strategy_name=self.setting.perception_type.get(),
                localization_strategy_name=self.setting.localization_type.get(),
                mapping_strategy_name=self.setting.mapping_type.get(),
                global_planner_strategy_name=self.setting.global_planner_type.get(),
                local_planner_strategy_name=self.setting.local_planner_type.get(),
                controller_strategy_name=self.setting.controller_type.get(),
                execution_task_names=self.setting.execution_task_names(),
                perception_dt=self.setting.perception_dt.get(),
                localization_dt=self.setting.localization_dt.get(),
                replan_dt=self.setting.replan_dt.get(),
                control_dt=self.setting.control_dt.get(),
                map_file=ExecutionSettings.c40_map,
                default_global_trajectory_file=self.setting.default_global_plan_file.get(),
                load_plugins=self.setting.c62_load_plugins.get(),
                async_combined_perception_planning=ExecutionSettings.c40_async_combined_perception_planning,
            )
        except Exception as e:
            error = e
            log.error("Error reloading stack: %s", e, exc_info=True)
        finally:
            self.local_plot.reset()
            self.global_plot.reset()
            self.stack_panels.refresh()
            self.update_views()
            if self.exec is not None:
                self.update_ui()
        if error is not None:
            self.QtWidgets.QMessageBox.critical(self.win, "Reload failed", str(error))

    def switch_profile(self) -> None:
        self.stack_panels.execution.stop_exec()
        nxt = self.setting.p60_next_profile.get() or "default"
        self.load_settings(profile=nxt)
        self.reload_stack(reload_code=False)
        self.update_ui()


def launch(*, offscreen: bool = False) -> int:
    """Create the visualizer and run the Qt event loop."""
    configure_qt_env(offscreen=offscreen)
    window = VisualizerQtWindow()
    return window.exec_app()
