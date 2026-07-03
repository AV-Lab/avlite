from __future__ import annotations

import logging
from typing import Any, ClassVar

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
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_apps.c52_factory import StackSettingsSync, get_stack_settings_classes as get_core_stack_settings_classes
from avlite.c50_apps.c53_plugins import load_plugin_settings_class, patch_plugin_settings
from avlite.c50_apps.c54_settings_schema import SettingsSchema
from avlite.c50_apps.c55_setting_utils import save_setting
from avlite.c50_apps.c58_paths import PluginPaths
from avlite.c50_apps.c59_settings import AppSettings, AppSettingsSchema
from avlite.plugins.p50_visualizer_tk.p53_ui_lib import DataPicker, TkSettingsBinder

log = logging.getLogger(__name__)


def _strategy_default(registry: dict, configured: str | None) -> str | None:
    if configured:
        return configured
    keys = list(registry.keys()) if registry else []
    return keys[0] if keys else None


class PluginSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/plugin_p50_visualizer_tk.yaml"

    p50_dark_mode: bool = Field(default=True, description="Use dark UI theme.")
    p50_hide_menubar: bool = Field(default=False, description="Hide the application menu bar.")
    p50_shortcut_mode: bool = Field(default=False, description="Enable keyboard shortcut mode in the visualizer.")
    p50_next_profile: str = Field(default="default", description="Profile to switch to with shortcut F.")
    p50_bg_color: str = Field(default="#333333", description="UI background color.")
    p50_fg_color: str = Field(default="white", description="UI foreground/text color.")
    p59_mouse_drag_slowdown_factor: float = Field(
        default=0.5, description="Slowdown factor when dragging plots with mouse."
    )

    p55_show_legend: bool = Field(default=False, description="Show plot legend (may reduce performance).")
    p55_show_past_locations: bool = Field(default=True, description="Show historical ego positions on plots.")
    p55_show_global_plan: bool = Field(default=True, description="Draw global plan on plots.")
    p55_show_global_plan_boundaries: bool = Field(
        default=True, description="Show left/right plan boundaries in global plot view."
    )
    p55_global_plan_velocity_scale: str = Field(
        default="relative",
        description="Global plan velocity color scale: 'relative' (per-path min–max) or 'absolute' (0 to ego max m/s).",
    )
    p55_show_local_plan: bool = Field(default=True, description="Draw local plan on plots.")
    p55_show_local_lattice: bool = Field(default=True, description="Draw local lattice on plots.")
    p55_show_state: bool = Field(default=True, description="Show ego state overlay on plots.")
    p55_global_view_follow_planner: bool = Field(default=False, description="Follow planner in global XY view.")
    p55_frenet_view_follow_planner: bool = Field(default=False, description="Follow planner in Frenet view.")
    p55_show_local_global_view: bool = Field(default=True, description="Show local global (XY) sub-view.")
    p55_show_local_frenet_view: bool = Field(default=True, description="Show local Frenet sub-view.")
    p55_show_lidar_global: bool = Field(default=True, description="Show LiDAR points in global XY view.")
    p55_show_lidar_frenet: bool = Field(default=True, description="Show LiDAR points in Frenet view.")
    p55_show_lidar_clusters: bool = Field(default=True, description="Highlight clustered LiDAR points.")
    p55_show_race_boundary: bool = Field(default=True, description="Show race boundary on plots.")
    p55_xy_zoom: float = Field(default=30, description="XY plot zoom level.")
    p55_frenet_zoom: float = Field(default=30, description="Frenet plot zoom level.")
    p55_global_zoom: float = Field(default=30, description="Global plot zoom level.")
    p55_show_occupancy_flow: bool = Field(default=False, description="Show occupancy flow visualization.")
    p55_show_perception_extras: bool = Field(default=False, description="Show extra perception debug overlays.")
    p55_global_plan_view: bool = Field(default=False, description="Show global plan panel.")
    p55_local_plan_view: bool = Field(default=False, description="Show local plan panel.")

    p57_show_core_logs: bool = Field(default=True, description="Show core module logs.")
    p57_show_perceive_logs: bool = Field(default=True, description="Show perception logs.")
    p57_show_plan_logs: bool = Field(default=True, description="Show planning logs.")
    p57_show_control_logs: bool = Field(default=True, description="Show control logs.")
    p57_show_execute_logs: bool = Field(default=True, description="Show execution logs.")
    p57_show_vis_logs: bool = Field(default=True, description="Show visualization logs.")
    p57_show_common_logs: bool = Field(default=True, description="Show common module logs.")
    p57_show_plugins_logs: bool = Field(default=True, description="Show plugin logs.")
    p57_disable_log: bool = Field(default=False, description="Disable log panel updates.")
    p57_max_log_lines: int = Field(default=1000, description="Max log lines retained in UI.")
    p57_log_view_expanded: bool = Field(default=False, description="Use expanded log panel height.")
    p57_log_view_default_height: int = Field(default=12, description="Default log panel height in lines.")
    p57_log_view_expended_height: int = Field(default=35, description="Expanded log panel height in lines.")
    p57_log_font: str = Field(default="Courier", description="Log panel font family.")
    p57_log_font_size: int = Field(default=11, description="Log panel font size.")
    p57_log_pull_time: int = Field(default=50, description="Log refresh interval (ms).")


PluginSettings = PluginSettingsSchema()


class AppSettingsUI:
    """Tk variables for AppSettings fields bound to widgets."""

    schema = AppSettingsSchema
    exclude = ["exclude", "filepath", "schema", "c50_default_plugins", "c50_community_plugins"]
    filepath: str = "configs/c59_apps.yaml"

    def __init__(self) -> None:
        self.c50_load_plugins = tk.BooleanVar(value=AppSettings.c50_load_plugins)
        self.c50_selected_profile = tk.StringVar(value=AppSettings.c50_selected_profile)

    def sync_from_singleton(self) -> None:
        """Copy persisted singleton values into Tk variables."""
        self.c50_load_plugins.set(AppSettings.c50_load_plugins)
        self.c50_selected_profile.set(AppSettings.c50_selected_profile)

    def sync_to_singleton(self) -> None:
        """Copy Tk variable values into the persisted singleton."""
        AppSettings.c50_load_plugins = bool(self.c50_load_plugins.get())
        AppSettings.c50_selected_profile = self.c50_selected_profile.get()




def get_stack_settings_classes() -> list[Any]:
    """Core stack settings plus visualization schema, for profile export/import."""
    return get_core_stack_settings_classes() + [PluginSettingsSchema()]


def _sync_exec_dt(attr: str, value: float) -> None:
    """Persist dt change to the ROS plugin YAML so it takes effect on next launch."""
    try:
        name = "avlite-executer-ROS2"
        stored = AppSettings.c50_community_plugins.get(name, name)
        install = str(PluginPaths.resolve(name, stored))
        cls = load_plugin_settings_class(name, install)
        if cls is None:
            return
        patch_plugin_settings(cls, name, install)
        setattr(cls, attr, float(value))
        save_setting(cls, binder=TkSettingsBinder())
    except Exception:
        pass


def sync_stack_settings_to_ui(setting: "VisualizationSettings") -> None:
    """Push stack singleton values into main-UI Tk vars without write-back."""
    setting.sync_stack_from_singletons()


def sync_perception_pipeline_from_c19(setting: "VisualizationSettings") -> None:
    """Push c19 pipeline strategy names into main-UI Tk vars without write-back."""
    setting.sync_perception_pipeline_from_c19()


class VisualizationSettings:
    """Runtime Tk variables for the visualizer; persisted via ``TkSettingsBinder``."""

    schema = PluginSettingsSchema
    exclude = [
        "exclude", "filepath", "schema", "vehicle_state", "elapsed_real_time",
        "elapsed_sim_time", "lap", "replan_fps", "control_fps", "perception_fps",
        "current_wp", "exec_running", "profile_list", "perception_status_text", "plugin_list",
        "detection_strategy_type", "tracking_strategy_type", "prediction_strategy_type",
        "_syncing_perception_pipeline", "_syncing_stack",
        "perception_type", "perception_dt", "localization_type", "localization_dt",
        "mapping_type", "global_planner_type", "local_planner_type", "controller_type",
        "executer_type", "exec_plan", "exec_control", "exec_perceive", "exec_localize",
        "control_dt", "replan_dt", "sim_dt", "execution_bridge",
        "default_global_plan_file", "default_map_file",
        "bridge_provide_ground_truth_detection", "bridge_provide_rgb_image",
        "bridge_provide_depth_image", "bridge_provide_lidar_data",
        "log_level", "log_to_file",
    ]
    filepath: str = "configs/plugin_p50_visualizer_tk.yaml"

    def __init__(self):
        self.p50_shortcut_mode = tk.BooleanVar()
        self.p50_next_profile = tk.StringVar(value=PluginSettings.p50_next_profile)
        self.p50_dark_mode = tk.BooleanVar(value=True)
        self.p50_hide_menubar = tk.BooleanVar(value=False)
        self.p59_mouse_drag_slowdown_factor = 0.5

        self.p55_show_legend = tk.BooleanVar(value=False)
        self.p55_show_past_locations = tk.BooleanVar(value=True)
        self.p55_show_global_plan = tk.BooleanVar(value=True)
        self.p55_show_global_plan_boundaries = tk.BooleanVar(value=True)
        self.p55_global_plan_velocity_scale = tk.StringVar(value="relative")
        self.p55_show_local_plan = tk.BooleanVar(value=True)
        self.p55_show_local_lattice = tk.BooleanVar(value=True)
        self.p55_show_state = tk.BooleanVar(value=True)
        self.p55_global_view_follow_planner = tk.BooleanVar(value=False)
        self.p55_frenet_view_follow_planner = tk.BooleanVar(value=False)
        self.p55_show_local_global_view = tk.BooleanVar(value=True)
        self.p55_show_local_frenet_view = tk.BooleanVar(value=True)
        self.p55_show_lidar_global = tk.BooleanVar(value=True)
        self.p55_show_lidar_frenet = tk.BooleanVar(value=False)
        self.p55_show_lidar_clusters = tk.BooleanVar(value=True)
        self.p55_show_race_boundary = tk.BooleanVar(value=True)

        self.p55_xy_zoom = 30
        self.p55_frenet_zoom = 30
        self.p55_global_zoom = 30

        self.p55_show_occupancy_flow = tk.BooleanVar(value=False)
        self.p55_show_perception_extras = tk.BooleanVar(value=False)
        self.vehicle_state = tk.StringVar(value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.perception_status_text = tk.StringVar(value="Spawn Agent: Right click on the plot.")

        self.perception_type = tk.StringVar(
            value=_strategy_default(PerceptionStrategy.registry, ExecutionSettings.c40_perception)
        )

        def _on_perception_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_perception = self.perception_type.get()

        self.perception_type.trace_add("write", _on_perception_change)

        self._syncing_stack = False
        self._syncing_perception_pipeline = False
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

        self.global_planner_type = tk.StringVar(
            value=ExecutionSettings.c40_global_planner
            if ExecutionSettings.c40_global_planner
            else (list(GlobalPlannerStrategy.registry.keys())[0] if GlobalPlannerStrategy.registry else None)
        )

        def _on_global_plan_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_global_planner = self.global_planner_type.get()

        self.global_planner_type.trace_add("write", _on_global_plan_change)

        self.local_planner_type = tk.StringVar(
            value=_strategy_default(LocalPlanningStrategy.registry, ExecutionSettings.c40_local_planner)
        )

        def _on_local_plan_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_local_planner = self.local_planner_type.get()

        self.local_planner_type.trace_add("write", _on_local_plan_change)

        self.lap = tk.StringVar(value="0")
        self.current_wp = tk.StringVar(value="0")

        self.controller_type = tk.StringVar(
            value=_strategy_default(ControlStrategy.registry, ExecutionSettings.c40_controller)
        )

        def _on_controller_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_controller = self.controller_type.get()

        self.controller_type.trace_add("write", _on_controller_change)

        self.p55_global_plan_view = tk.BooleanVar(value=False)
        self.p55_local_plan_view = tk.BooleanVar(value=False)

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
            _sync_exec_dt("control_dt", self.control_dt.get())

        self.control_dt.trace_add("write", _on_control_dt_change)

        self.replan_dt = tk.DoubleVar(value=ExecutionSettings.c40_replan_dt)

        def _on_replan_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_replan_dt = float(self.replan_dt.get())
            _sync_exec_dt("replan_dt", self.replan_dt.get())

        self.replan_dt.trace_add("write", _on_replan_dt_change)

        self.perception_dt = tk.DoubleVar(value=ExecutionSettings.c40_perception_dt)

        def _on_perception_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_perception_dt = float(self.perception_dt.get())
            _sync_exec_dt("perception_dt", self.perception_dt.get())

        self.perception_dt.trace_add("write", _on_perception_dt_change)

        self.sim_dt = tk.DoubleVar(value=ExecutionSettings.c40_sim_dt)

        def _on_sim_dt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_sim_dt = float(self.sim_dt.get())

        self.sim_dt.trace_add("write", _on_sim_dt_change)

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

        self.bridge_provide_ground_truth_detection = tk.BooleanVar(value=ExecutionSettings.c41_provide_ground_truth)

        def _on_gt_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c41_provide_ground_truth = self.bridge_provide_ground_truth_detection.get()

        self.bridge_provide_ground_truth_detection.trace_add("write", _on_gt_change)

        self.bridge_provide_rgb_image = tk.BooleanVar(value=ExecutionSettings.c41_provide_rgb)

        def _on_rgb_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c41_provide_rgb = self.bridge_provide_rgb_image.get()

        self.bridge_provide_rgb_image.trace_add("write", _on_rgb_change)
        self.bridge_provide_depth_image = tk.BooleanVar(value=ExecutionSettings.c41_provide_depth)
        self.bridge_provide_lidar_data = tk.BooleanVar(value=ExecutionSettings.c41_provide_lidar)

        def _on_lidar_change(*args):
            if self._syncing_stack:
                return
            ExecutionSettings.c41_provide_lidar = self.bridge_provide_lidar_data.get()

        self.bridge_provide_lidar_data.trace_add("write", _on_lidar_change)

        self.log_level = tk.StringVar(value=ExecutionSettings.c40_log_level)

        def _on_log_level_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_log_level = self.log_level.get()

        self.log_level.trace_add("write", _on_log_level_change)
        self.p57_show_core_logs = tk.BooleanVar(value=True)
        self.p57_show_perceive_logs = tk.BooleanVar(value=True)
        self.p57_show_plan_logs = tk.BooleanVar(value=True)
        self.p57_show_control_logs = tk.BooleanVar(value=True)
        self.p57_show_execute_logs = tk.BooleanVar(value=True)
        self.p57_show_vis_logs = tk.BooleanVar(value=True)
        self.p57_show_common_logs = tk.BooleanVar(value=True)
        self.p57_show_plugins_logs = tk.BooleanVar(value=True)
        self.p57_disable_log = tk.BooleanVar(value=False)
        self.p57_max_log_lines = 1000
        self.p57_log_view_expanded = tk.BooleanVar(value=False)
        self.p57_log_view_default_height = tk.IntVar(value=12)
        self.p57_log_view_expended_height = tk.IntVar(value=35)
        self.p57_log_font = tk.StringVar(value="Courier")
        self.p57_log_font_size = tk.IntVar(value=11)
        self.log_to_file = tk.BooleanVar(value=ExecutionSettings.c40_log_to_file)

        def _on_log_to_file_change(*_):
            if self._syncing_stack:
                return
            ExecutionSettings.c40_log_to_file = self.log_to_file.get()

        self.log_to_file.trace_add("write", _on_log_to_file_change)
        self.p57_log_pull_time = 50
        self.p50_bg_color = "#333333" if self.p50_dark_mode.get() else "white"
        self.p50_fg_color = "white" if self.p50_dark_mode.get() else "black"
        self.profile_list = []

    def sync_perception_pipeline_from_c19(self) -> None:
        """Push c19 pipeline strategy names into main-UI Tk vars without write-back."""
        self._syncing_perception_pipeline = True
        try:
            self.detection_strategy_type.set(PerceptionSettings.c12_detection_strategy)
            self.tracking_strategy_type.set(PerceptionSettings.c12_tracking_strategy)
            self.prediction_strategy_type.set(PerceptionSettings.c12_prediction_strategy)
        finally:
            self._syncing_perception_pipeline = False

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
            self.global_planner_type.set(
                es.c40_global_planner
                or _strategy_default(GlobalPlannerStrategy.registry, None)
                or ""
            )
            self.local_planner_type.set(
                es.c40_local_planner
                or _strategy_default(LocalPlanningStrategy.registry, None)
                or ""
            )
            self.controller_type.set(
                es.c40_controller or _strategy_default(ControlStrategy.registry, None) or ""
            )
            self.executer_type.set(
                es.c40_executer_type or _strategy_default(ExecutionStrategy.registry, None) or ""
            )
            self.control_dt.set(es.c40_control_dt)
            self.replan_dt.set(es.c40_replan_dt)
            self.sim_dt.set(es.c40_sim_dt)
            self.execution_bridge.set(es.c40_bridge)
            self.bridge_provide_ground_truth_detection.set(es.c41_provide_ground_truth)
            self.bridge_provide_rgb_image.set(es.c41_provide_rgb)
            self.bridge_provide_depth_image.set(es.c41_provide_depth)
            self.bridge_provide_lidar_data.set(es.c41_provide_lidar)
            self.log_level.set(es.c40_log_level)
            self.log_to_file.set(es.c40_log_to_file)
            self.sync_perception_pipeline_from_c19()
        finally:
            self._syncing_stack = False
