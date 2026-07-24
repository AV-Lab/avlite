from __future__ import annotations

import logging

import tkinter as tk
from pydantic import Field

from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PerceptionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import StackSettingsSync
from avlite.c60_apps.c63_plugins import load_plugin_settings_class, patch_plugin_settings
from avlite.c60_apps.c64_settings_schema import SettingsSchema
from avlite.c60_apps.c65_setting_utils import save_setting
from avlite.c60_apps.c68_paths import PluginPaths
from avlite.c60_apps.c69_settings import AppSettings
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import DataPicker, TkSettingsBinder

log = logging.getLogger(__name__)


def _strategy_default(registry: dict, configured: str | None) -> str | None:
    if configured:
        return configured
    keys = list(registry.keys()) if registry else []
    return keys[0] if keys else None


class PluginSettingsSchema(SettingsSchema):
    p60_dark_mode: bool = Field(default=True, description="Use dark UI theme.")
    p60_hide_menubar: bool = Field(default=False, description="Hide the application menu bar.")
    p60_shortcut_mode: bool = Field(default=False, description="Enable keyboard shortcut mode in the visualizer.")
    p60_next_profile: str = Field(default="default", description="Profile to switch to with shortcut F.")
    p60_bg_color: str = Field(default="#333333", description="UI background color.")
    p60_fg_color: str = Field(default="white", description="UI foreground/text color.")
    p69_mouse_drag_slowdown_factor: float = Field(
        default=0.5, description="Slowdown factor when dragging plots with mouse."
    )

    p66_show_legend: bool = Field(default=False, description="Show plot legend (may reduce performance).")
    p66_show_past_locations: bool = Field(default=True, description="Show historical ego positions on plots.")
    p66_show_global_plan: bool = Field(default=True, description="Draw global plan on plots.")
    p66_show_global_plan_boundaries: bool = Field(
        default=True, description="Show left/right plan boundaries in global plot view."
    )
    p66_global_plan_velocity_scale: str = Field(
        default="relative",
        description="Global plan velocity color scale: 'relative' (per-path min–max) or 'absolute' (0 to ego max m/s).",
    )
    p66_show_local_plan: bool = Field(default=True, description="Draw local plan on plots.")
    p66_show_local_lattice: bool = Field(default=True, description="Draw local lattice on plots.")
    p66_show_state: bool = Field(default=True, description="Show ego state overlay on plots.")
    p66_global_view_follow_planner: bool = Field(default=False, description="Follow planner in global XY view.")
    p66_frenet_view_follow_planner: bool = Field(default=False, description="Follow planner in Frenet view.")
    p67_show_local_global_view: bool = Field(default=True, description="Show local global (XY) sub-view.")
    p67_show_local_frenet_view: bool = Field(default=True, description="Show local Frenet sub-view.")
    p66_show_lidar_global: bool = Field(default=True, description="Show LiDAR points in global XY view.")
    p66_show_lidar_frenet: bool = Field(default=True, description="Show LiDAR points in Frenet view.")
    p66_show_lidar_clusters: bool = Field(default=True, description="Highlight clustered LiDAR points.")
    p66_show_race_boundary: bool = Field(default=True, description="Show race boundary on plots.")
    p66_xy_zoom: float = Field(default=30, description="XY plot zoom level.")
    p66_frenet_zoom: float = Field(default=30, description="Frenet plot zoom level.")
    p66_global_zoom: float = Field(default=30, description="Global plot zoom level.")
    p67_show_occupancy_flow: bool = Field(default=False, description="Show occupancy flow visualization.")
    p67_show_perception_extras: bool = Field(default=False, description="Show extra perception debug overlays.")
    p67_show_prediction: bool = Field(default=True, description="Show predicted agent trajectories on plots.")
    p67_global_plan_view: bool = Field(default=True, description="Show global plan panel.")
    p67_local_plan_view: bool = Field(default=True, description="Show local plan panel.")

    p68_show_core_logs: bool = Field(default=True, description="Show core module logs.")
    p68_show_perceive_logs: bool = Field(default=True, description="Show perception logs.")
    p68_show_plan_logs: bool = Field(default=True, description="Show planning logs.")
    p68_show_control_logs: bool = Field(default=True, description="Show control logs.")
    p68_show_execute_logs: bool = Field(default=True, description="Show execution logs.")
    p68_show_vis_logs: bool = Field(default=True, description="Show visualization logs.")
    p68_show_common_logs: bool = Field(default=True, description="Show common module logs.")
    p68_show_plugins_logs: bool = Field(default=True, description="Show plugin logs.")
    p68_disable_log: bool = Field(default=False, description="Disable log panel updates.")
    p68_max_log_lines: int = Field(default=1000, description="Max log lines retained in UI.")
    p68_log_view_expanded: bool = Field(default=False, description="Use expanded log panel height.")
    p68_log_view_default_height: int = Field(default=12, description="Default log panel height in lines.")
    p68_log_view_expended_height: int = Field(default=35, description="Expanded log panel height in lines.")
    p68_log_font: str = Field(default="Courier", description="Log panel font family.")
    p68_log_font_size: int = Field(default=11, description="Log panel font size.")
    p68_log_pull_time: int = Field(default=50, description="Log refresh interval (ms).")


PluginSettings = PluginSettingsSchema()


def sync_stack_settings_to_ui(setting: "VisualizationSettings") -> None:
    """Push stack singleton values into main-UI Tk vars without write-back."""
    setting.sync_stack_from_singletons()


class VisualizationSettings:
    """Runtime Tk variables for the visualizer; persisted via ``TkSettingsBinder``."""

    schema = PluginSettingsSchema

    def __init__(self):
        self.c62_load_plugins = tk.BooleanVar(value=AppSettings.c62_load_plugins)
        self.c60_selected_profile = tk.StringVar(value=AppSettings.c60_selected_profile)
        self.p60_shortcut_mode = tk.BooleanVar()
        self.p60_next_profile = tk.StringVar(value=PluginSettings.p60_next_profile)
        self.p60_dark_mode = tk.BooleanVar(value=True)
        self.p60_hide_menubar = tk.BooleanVar(value=False)
        self.p69_mouse_drag_slowdown_factor = PluginSettings.p69_mouse_drag_slowdown_factor

        self.p66_show_legend = tk.BooleanVar(value=False)
        self.p66_show_past_locations = tk.BooleanVar(value=True)
        self.p66_show_global_plan = tk.BooleanVar(value=True)
        self.p66_show_global_plan_boundaries = tk.BooleanVar(value=True)
        self.p66_global_plan_velocity_scale = tk.StringVar(value="relative")
        self.p66_show_local_plan = tk.BooleanVar(value=True)
        self.p66_show_local_lattice = tk.BooleanVar(value=True)
        self.p66_show_state = tk.BooleanVar(value=True)
        self.p66_global_view_follow_planner = tk.BooleanVar(value=False)
        self.p66_frenet_view_follow_planner = tk.BooleanVar(value=False)
        self.p67_show_local_global_view = tk.BooleanVar(value=True)
        self.p67_show_local_frenet_view = tk.BooleanVar(value=True)
        self.p66_show_lidar_global = tk.BooleanVar(value=True)
        self.p66_show_lidar_frenet = tk.BooleanVar(value=False)
        self.p66_show_lidar_clusters = tk.BooleanVar(value=True)
        self.p66_show_race_boundary = tk.BooleanVar(value=True)

        self.p66_xy_zoom = 30
        self.p66_frenet_zoom = 30
        self.p66_global_zoom = 30

        self.p67_show_occupancy_flow = tk.BooleanVar(value=False)
        self.p67_show_perception_extras = tk.BooleanVar(value=False)
        self.p67_show_prediction = tk.BooleanVar(value=True)
        self.vehicle_state = tk.StringVar(value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.perception_status_text = tk.StringVar(value="R-click: spawn")

        self.perception_type = tk.StringVar(value=ExecutionSettings.c40_perception or "")

        def _on_perception_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_perception = self.perception_type.get()

        self.perception_type.trace_add("write", _on_perception_change)

        self._syncing_stack = False
        self._syncing_perception_pipeline = False
        self._syncing_local_planning_pipeline = False
        self.detection_strategy_type = tk.StringVar(value=PerceptionSettings.c12_detection_strategy)

        def _on_detection_change(*args):
            if self._syncing_stack or self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_detection_strategy = self.detection_strategy_type.get()

        self.detection_strategy_type.trace_add("write", _on_detection_change)

        self.tracking_strategy_type = tk.StringVar(value=PerceptionSettings.c12_tracking_strategy)

        def _on_tracking_change(*args):
            if self._syncing_stack or self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_tracking_strategy = self.tracking_strategy_type.get()

        self.tracking_strategy_type.trace_add("write", _on_tracking_change)

        self.prediction_strategy_type = tk.StringVar(value=PerceptionSettings.c12_prediction_strategy)

        def _on_prediction_change(*args):
            if self._syncing_stack or self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_prediction_strategy = self.prediction_strategy_type.get()

        self.prediction_strategy_type.trace_add("write", _on_prediction_change)

        self.localization_type = tk.StringVar(value=ExecutionSettings.c40_localization)

        def _on_localization_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_localization = self.localization_type.get()

        self.localization_type.trace_add("write", _on_localization_change)
        self.localization_dt = tk.DoubleVar(value=ExecutionSettings.c40_localization_dt)

        def _on_localization_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_localization_dt = float(self.localization_dt.get())

        self.localization_dt.trace_add("write", _on_localization_dt_change)

        self.mapping_type = tk.StringVar(value=ExecutionSettings.c40_mapping)

        def _on_mapping_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_mapping = self.mapping_type.get()

        self.mapping_type.trace_add("write", _on_mapping_change)

        self.global_planner_type = tk.StringVar(value=ExecutionSettings.c40_global_planner or "")

        def _on_global_plan_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_global_planner = self.global_planner_type.get()

        self.global_planner_type.trace_add("write", _on_global_plan_change)

        self.local_planner_type = tk.StringVar(value=ExecutionSettings.c40_local_planner or "")

        def _on_local_plan_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_local_planner = self.local_planner_type.get()

        self.local_planner_type.trace_add("write", _on_local_plan_change)

        self.behavioral_strategy_type = tk.StringVar(value=PlanningSettings.c23_behavioral_strategy)

        def _on_behavioral_change(*args):
            if self._syncing_stack or self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_behavioral_strategy = self.behavioral_strategy_type.get()

        self.behavioral_strategy_type.trace_add("write", _on_behavioral_change)

        self.path_strategy_type = tk.StringVar(value=PlanningSettings.c23_path_strategy)

        def _on_path_change(*args):
            if self._syncing_stack or self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_path_strategy = self.path_strategy_type.get()

        self.path_strategy_type.trace_add("write", _on_path_change)

        self.velocity_strategy_type = tk.StringVar(value=PlanningSettings.c23_velocity_strategy)

        def _on_velocity_change(*args):
            if self._syncing_stack or self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_velocity_strategy = self.velocity_strategy_type.get()

        self.velocity_strategy_type.trace_add("write", _on_velocity_change)

        self.lap = tk.StringVar(value="0")
        self.current_wp = tk.StringVar(value="0")

        self.controller_type = tk.StringVar(value=ExecutionSettings.c40_controller or "")

        def _on_controller_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_controller = self.controller_type.get()

        self.controller_type.trace_add("write", _on_controller_change)

        self.execution_tasks = tk.StringVar(
            value=",".join(ExecutionSettings.c40_execution_tasks or [])
        )

        def _on_execution_tasks_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_execution_tasks = list(self.execution_task_names())

        self.execution_tasks.trace_add("write", _on_execution_tasks_change)

        self.p67_global_plan_view = tk.BooleanVar(value=True)
        self.p67_local_plan_view = tk.BooleanVar(value=True)

        self.executer_type = tk.StringVar(
            value=_strategy_default(ExecutionStrategy.registry, ExecutionSettings.c40_executer_type)
        )

        def _on_executer_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_executer_type = self.executer_type.get()

        self.executer_type.trace_add("write", _on_executer_change)

        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)
        self.exec_localize = tk.BooleanVar(value=True)
        self.exec_running = False

        self.control_dt = tk.DoubleVar(value=ExecutionSettings.c40_control_dt)

        def _on_control_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_control_dt = float(self.control_dt.get())
            self._sync_exec_dt("control_dt", self.control_dt.get())

        self.control_dt.trace_add("write", _on_control_dt_change)

        self.replan_dt = tk.DoubleVar(value=ExecutionSettings.c40_replan_dt)

        def _on_replan_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_replan_dt = float(self.replan_dt.get())
            self._sync_exec_dt("replan_dt", self.replan_dt.get())

        self.replan_dt.trace_add("write", _on_replan_dt_change)

        self.perception_dt = tk.DoubleVar(value=ExecutionSettings.c40_perception_dt)

        def _on_perception_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_perception_dt = float(self.perception_dt.get())
            self._sync_exec_dt("perception_dt", self.perception_dt.get())

        self.perception_dt.trace_add("write", _on_perception_dt_change)

        self.sim_dt = tk.DoubleVar(value=ExecutionSettings.c40_sim_dt)

        def _on_sim_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_sim_dt = float(self.sim_dt.get())
            self._sync_exec_dt("sim_dt", self.sim_dt.get())

        self.sim_dt.trace_add("write", _on_sim_dt_change)

        self.pace_perception = tk.BooleanVar(value=ExecutionSettings.c40_pace_perception)
        self.pace_replan = tk.BooleanVar(value=ExecutionSettings.c40_pace_replan)
        self.pace_control = tk.BooleanVar(value=ExecutionSettings.c40_pace_control)
        self.pace_sim = tk.BooleanVar(value=ExecutionSettings.c40_pace_sim)

        def _on_pace_perception_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_pace_perception = bool(self.pace_perception.get())
            self._sync_exec_pace("pace_perception", self.pace_perception.get())

        def _on_pace_replan_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_pace_replan = bool(self.pace_replan.get())
            self._sync_exec_pace("pace_replan", self.pace_replan.get())

        def _on_pace_control_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_pace_control = bool(self.pace_control.get())
            self._sync_exec_pace("pace_control", self.pace_control.get())

        def _on_pace_sim_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_pace_sim = bool(self.pace_sim.get())
            self._sync_exec_pace("pace_sim", self.pace_sim.get())

        self.pace_perception.trace_add("write", _on_pace_perception_change)
        self.pace_replan.trace_add("write", _on_pace_replan_change)
        self.pace_control.trace_add("write", _on_pace_control_change)
        self.pace_sim.trace_add("write", _on_pace_sim_change)

        self.execution_bridge = tk.StringVar(value=ExecutionSettings.c40_bridge)

        def _on_execution_bridge_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_bridge = self.execution_bridge.get()

        self.execution_bridge.trace_add("write", _on_execution_bridge_change)

        self.default_global_plan_file = tk.StringVar(value=DataPicker.default_global_plan_display_path())

        def _on_default_global_plan_file_change(*args):
            StackSettingsSync.apply_global_plan_selection(self.default_global_plan_file.get())

        self.default_global_plan_file.trace_add("write", _on_default_global_plan_file_change)

        self.default_map_file = tk.StringVar(value=DataPicker.default_map_display_path())

        def _on_default_map_file_change(*args):
            StackSettingsSync.apply_map_selection(self.default_map_file.get())

        self.default_map_file.trace_add("write", _on_default_map_file_change)

        self.elapsed_real_time = tk.StringVar(value="0")
        self.elapsed_sim_time = tk.StringVar(value="0")
        self.replan_fps = tk.StringVar(value="0")
        self.control_fps = tk.StringVar(value="0")
        self.perception_fps = tk.StringVar(value="0")

        self.log_level = tk.StringVar(value=ExecutionSettings.c40_log_level)

        def _on_log_level_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_log_level = self.log_level.get()

        self.log_level.trace_add("write", _on_log_level_change)
        self.p68_show_core_logs = tk.BooleanVar(value=True)
        self.p68_show_perceive_logs = tk.BooleanVar(value=True)
        self.p68_show_plan_logs = tk.BooleanVar(value=True)
        self.p68_show_control_logs = tk.BooleanVar(value=True)
        self.p68_show_execute_logs = tk.BooleanVar(value=True)
        self.p68_show_vis_logs = tk.BooleanVar(value=True)
        self.p68_show_common_logs = tk.BooleanVar(value=True)
        self.p68_show_plugins_logs = tk.BooleanVar(value=True)
        self.p68_disable_log = tk.BooleanVar(value=False)
        self.p68_max_log_lines = 1000
        self.p68_log_view_expanded = tk.BooleanVar(value=False)
        self.p68_log_view_default_height = tk.IntVar(value=12)
        self.p68_log_view_expended_height = tk.IntVar(value=35)
        self.p68_log_font = tk.StringVar(value="Courier")
        self.p68_log_font_size = tk.IntVar(value=11)
        self.log_to_file = tk.BooleanVar(value=ExecutionSettings.c40_log_to_file)

        def _on_log_to_file_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_log_to_file = self.log_to_file.get()

        self.log_to_file.trace_add("write", _on_log_to_file_change)
        self.p68_log_pull_time = 50
        self.p60_bg_color = "#333333" if self.p60_dark_mode.get() else "white"
        self.p60_fg_color = "white" if self.p60_dark_mode.get() else "black"
        self.profile_list = []

    def _sync_exec_dt(self, attr: str, value: float) -> None:
        """Persist dt change to the ROS plugin YAML so it takes effect on next launch."""
        self._sync_ros_plugin_attr(attr, float(value))

    def _sync_exec_pace(self, attr: str, value: bool) -> None:
        """Persist pace flag to the ROS plugin YAML so it takes effect on next launch."""
        self._sync_ros_plugin_attr(attr, bool(value))

    def _sync_ros_plugin_attr(self, attr: str, value) -> None:
        try:
            name = "avlite-executer-ROS2"
            stored = AppSettings.c62_community_plugins.get(name, name)
            install = str(PluginPaths.resolve(name, stored))
            cls = load_plugin_settings_class(name, install)
            if cls is None or not hasattr(cls, attr):
                return
            patch_plugin_settings(cls, name, install)
            setattr(cls, attr, value)
            save_setting(cls, binder=TkSettingsBinder())
        except Exception:
            pass

    def sync_app_from_singleton(self) -> None:
        """Copy AppSettings singleton values into Tk variables."""
        self.c62_load_plugins.set(AppSettings.c62_load_plugins)
        self.c60_selected_profile.set(AppSettings.c60_selected_profile)

    def sync_app_to_singleton(self) -> None:
        """Copy app Tk variable values into the AppSettings singleton."""
        AppSettings.c62_load_plugins = bool(self.c62_load_plugins.get())
        AppSettings.c60_selected_profile = self.c60_selected_profile.get()

    def sync_perception_pipeline_from_c19(self) -> None:
        """Push c19 pipeline strategy names into main-UI Tk vars without write-back."""
        self._syncing_perception_pipeline = True
        try:
            self.detection_strategy_type.set(PerceptionSettings.c12_detection_strategy)
            self.tracking_strategy_type.set(PerceptionSettings.c12_tracking_strategy)
            self.prediction_strategy_type.set(PerceptionSettings.c12_prediction_strategy)
        finally:
            self._syncing_perception_pipeline = False

    def sync_local_planning_pipeline_from_c29(self) -> None:
        """Push c23 pipeline strategy names into main-UI Tk vars without write-back."""
        self._syncing_local_planning_pipeline = True
        try:
            self.behavioral_strategy_type.set(PlanningSettings.c23_behavioral_strategy)
            self.path_strategy_type.set(PlanningSettings.c23_path_strategy)
            self.velocity_strategy_type.set(PlanningSettings.c23_velocity_strategy)
        finally:
            self._syncing_local_planning_pipeline = False

    def execution_task_names(self) -> list[str]:
        """Parsed TaskStrategy class names from the comma-joined StringVar."""
        return [part.strip() for part in self.execution_tasks.get().split(",") if part.strip()]

    def sync_stack_from_singletons(self) -> None:
        """Push stack singleton values into Tk vars without write-back."""
        self._syncing_stack = True
        try:
            es = ExecutionSettings
            self.perception_type.set(es.c40_perception or "")
            self.perception_dt.set(es.c40_perception_dt)
            self.localization_type.set(es.c40_localization or "")
            self.localization_dt.set(es.c40_localization_dt)
            self.mapping_type.set(es.c40_mapping or "")
            self.global_planner_type.set(es.c40_global_planner or "")
            self.local_planner_type.set(es.c40_local_planner or "")
            self.controller_type.set(es.c40_controller or "")
            self.execution_tasks.set(",".join(es.c40_execution_tasks or []))
            self.executer_type.set(
                es.c40_executer_type or _strategy_default(ExecutionStrategy.registry, None) or ""
            )
            self.control_dt.set(es.c40_control_dt)
            self.replan_dt.set(es.c40_replan_dt)
            self.sim_dt.set(es.c40_sim_dt)
            self.pace_perception.set(es.c40_pace_perception)
            self.pace_replan.set(es.c40_pace_replan)
            self.pace_control.set(es.c40_pace_control)
            self.pace_sim.set(es.c40_pace_sim)
            self.execution_bridge.set(es.c40_bridge)
            self.log_level.set(es.c40_log_level)
            self.log_to_file.set(es.c40_log_to_file)
            self.sync_perception_pipeline_from_c19()
            self.sync_local_planning_pipeline_from_c29()
        finally:
            self._syncing_stack = False
