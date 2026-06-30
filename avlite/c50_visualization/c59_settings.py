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
from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c43_factory import StackSettingsSync
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_common.c68_settings_schema import SettingsSchema

log = logging.getLogger(__name__)
class VisualizationSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c50_visualization.yaml"

    shortcut_mode: bool = Field(default=False, description="Enable keyboard shortcut mode in the visualizer.")
    dark_mode: bool = Field(default=True, description="Use dark UI theme.")
    hide_menubar: bool = Field(default=False, description="Hide the application menu bar.")
    selected_profile: str = Field(default="default", description="Active settings profile name.")
    next_profile: str = Field(default="default", description="Profile to switch to with shortcut F.")
    load_plugins: bool = Field(default=True, description="Load built-in and community plugins on startup.")
    mouse_drag_slowdown_factor: float = Field(default=0.5, description="Slowdown factor when dragging plots with mouse.")

    show_legend: bool = Field(default=False, description="Show plot legend (may reduce performance).")
    show_past_locations: bool = Field(default=True, description="Show historical ego positions on plots.")
    show_global_plan: bool = Field(default=True, description="Draw global plan on plots.")
    show_local_plan: bool = Field(default=True, description="Draw local plan on plots.")
    show_local_lattice: bool = Field(default=True, description="Draw local lattice on plots.")
    show_state: bool = Field(default=True, description="Show ego state overlay on plots.")
    global_view_follow_planner: bool = Field(default=False, description="Follow planner in global XY view.")
    frenet_view_follow_planner: bool = Field(default=False, description="Follow planner in Frenet view.")
    show_local_global_view: bool = Field(default=True, description="Show local global (XY) sub-view.")
    show_local_frenet_view: bool = Field(default=True, description="Show local Frenet sub-view.")
    show_lidar_global: bool = Field(default=True, description="Show LiDAR points in global XY view.")
    show_lidar_frenet: bool = Field(default=False, description="Show LiDAR points in Frenet view.")
    show_lidar_clusters: bool = Field(default=True, description="Highlight clustered LiDAR points.")
    show_race_boundary: bool = Field(default=True, description="Show race boundary on plots.")
    xy_zoom: float = Field(default=30, description="XY plot zoom level.")
    frenet_zoom: float = Field(default=30, description="Frenet plot zoom level.")
    global_zoom: float = Field(default=30, description="Global plot zoom level.")

    show_occupancy_flow: bool = Field(default=False, description="Show occupancy flow visualization.")
    show_perception_extras: bool = Field(default=False, description="Show extra perception debug overlays.")
    perception_type: str = Field(default="", description="Selected perception strategy class name.")
    perception_dt: float = Field(default=0.5, description="UI-linked perception dt (seconds).")
    detection_strategy_type: str = Field(default="", description="Detection sub-strategy display name.")
    tracking_strategy_type: str = Field(default="", description="Tracking sub-strategy display name.")
    prediction_strategy_type: str = Field(default="", description="Prediction sub-strategy display name.")
    localization_type: str = Field(default="", description="Localization strategy display name.")
    localization_dt: float = Field(default=0.1, description="UI-linked localization dt (seconds).")
    mapping_type: str = Field(default="", description="Mapping strategy display name.")
    global_planner_type: str = Field(default="", description="Global planner class name.")
    local_planner_type: str = Field(default="", description="Local planner class name.")
    controller_type: str = Field(default="", description="Controller class name.")
    global_plan_view: bool = Field(default=False, description="Show global plan panel.")
    local_plan_view: bool = Field(default=False, description="Show local plan panel.")
    executer_type: str = Field(default="", description="Executer class name.")
    exec_plan: bool = Field(default=True, description="Enable planning step in execution.")
    exec_control: bool = Field(default=True, description="Enable control step in execution.")
    exec_perceive: bool = Field(default=True, description="Enable perception step in execution.")
    exec_localize: bool = Field(default=True, description="Enable localization step in execution.")
    control_dt: float = Field(default=0.01, description="UI-linked control dt (seconds).")
    replan_dt: float = Field(default=0.5, description="UI-linked replan dt (seconds).")
    sim_dt: float = Field(default=0.01, description="UI-linked simulation dt (seconds).")
    execution_bridge: str = Field(default="BasicSim", description="World bridge class name.")
    default_global_plan_file: str = Field(default="", description="Default global plan file path.")
    default_map_file: str = Field(default="", description="Default map file path (HD map or race boundary).")
    bridge_provide_ground_truth_detection: bool = Field(default=False, description="Request ground truth from bridge.")
    bridge_provide_rgb_image: bool = Field(default=False, description="Request RGB from bridge.")
    bridge_provide_depth_image: bool = Field(default=False, description="Request depth from bridge.")
    bridge_provide_lidar_data: bool = Field(default=False, description="Request LiDAR from bridge.")
    log_level: str = Field(default="INFO", description="UI logging level filter.")
    show_core_logs: bool = Field(default=True, description="Show core module logs.")
    show_perceive_logs: bool = Field(default=True, description="Show perception logs.")
    show_plan_logs: bool = Field(default=True, description="Show planning logs.")
    show_control_logs: bool = Field(default=True, description="Show control logs.")
    show_execute_logs: bool = Field(default=True, description="Show execution logs.")
    show_vis_logs: bool = Field(default=True, description="Show visualization logs.")
    show_common_logs: bool = Field(default=True, description="Show common module logs.")
    show_plugins_logs: bool = Field(default=True, description="Show plugin logs.")
    disable_log: bool = Field(default=False, description="Disable log panel updates.")
    max_log_lines: int = Field(default=1000, description="Max log lines retained in UI.")
    log_view_expanded: bool = Field(default=False, description="Use expanded log panel height.")
    log_view_default_height: int = Field(default=12, description="Default log panel height in lines.")
    log_view_expended_height: int = Field(default=35, description="Expanded log panel height in lines.")
    log_font: str = Field(default="Courier", description="Log panel font family.")
    log_font_size: int = Field(default=11, description="Log panel font size.")
    log_to_file: bool = Field(default=False, description="Write logs to file.")
    log_pull_time: int = Field(default=50, description="Log refresh interval (ms).")
    bg_color: str = Field(default="#333333", description="UI background color.")
    fg_color: str = Field(default="white", description="UI foreground/text color.")


def get_stack_settings_classes() -> list[Any]:
    """Core stack settings plus visualization schema, for profile export/import."""
    from avlite.c40_execution.c43_factory import get_stack_settings_classes as get_core_stack_settings_classes

    return get_core_stack_settings_classes() + [VisualizationSettingsSchema()]


def _sync_exec_dt(attr: str, value: float) -> None:
    """Persist dt change to the ROS plugin YAML so it takes effect on next launch."""
    try:
        from avlite.plugins.p40_executer_ROS2.settings import PluginSettings as ROSSettings
        from avlite.c50_visualization.c58_ui_lib import TkSettingsBinder
        from avlite.c60_common.c69_setting_utils import save_setting

        setattr(ROSSettings, attr, float(value))
        save_setting(ROSSettings, binder=TkSettingsBinder())
    except Exception:
        pass


class VisualizationSettings:
    """Runtime Tk variables for the visualizer; persisted via ``TkSettingsBinder``."""

    schema = VisualizationSettingsSchema
    exclude = [
        "exclude", "filepath", "schema", "vehicle_state", "elapsed_real_time",
        "elapsed_sim_time", "lap", "replan_fps", "control_fps", "perception_fps",
        "current_wp", "exec_running", "profile_list", "perception_status_text", "plugin_list",
    ]
    filepath: str = "configs/c50_visualization.yaml"

    def __init__(self):
        from avlite.c50_visualization.c58_ui_lib import DataPicker

        self.shortcut_mode = tk.BooleanVar()
        self.dark_mode = tk.BooleanVar(value=True)
        self.hide_menubar = tk.BooleanVar(value=False)
        self.selected_profile = tk.StringVar(value="default")
        self.next_profile = tk.StringVar(value="default")
        self.load_plugins = tk.BooleanVar(value=True)
        self.mouse_drag_slowdown_factor = 0.5

        self.show_legend = tk.BooleanVar(value=False)
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_global_plan = tk.BooleanVar(value=True)
        self.show_local_plan = tk.BooleanVar(value=True)
        self.show_local_lattice = tk.BooleanVar(value=True)
        self.show_state = tk.BooleanVar(value=True)
        self.global_view_follow_planner = tk.BooleanVar(value=False)
        self.frenet_view_follow_planner = tk.BooleanVar(value=False)
        self.show_local_global_view = tk.BooleanVar(value=True)
        self.show_local_frenet_view = tk.BooleanVar(value=True)
        self.show_lidar_global = tk.BooleanVar(value=True)
        self.show_lidar_frenet = tk.BooleanVar(value=False)
        self.show_lidar_clusters = tk.BooleanVar(value=True)
        self.show_race_boundary = tk.BooleanVar(value=True)

        self.xy_zoom = 30
        self.frenet_zoom = 30
        self.global_zoom = 30

        self.show_occupancy_flow = tk.BooleanVar(value=False)
        self.show_perception_extras = tk.BooleanVar(value=False)
        self.vehicle_state = tk.StringVar(value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.perception_status_text = tk.StringVar(value="Spawn Agent: Right click on the plot.")

        self.perception_type = tk.StringVar(
            value=list(PerceptionStrategy.registry.keys())[0] if PerceptionStrategy.registry else None
        )

        def _on_perception_change(*args):
            ExecutionSettings.c40_perception = self.perception_type.get()

        self.perception_type.trace_add("write", _on_perception_change)
        self.perception_dt = tk.DoubleVar(value=ExecutionSettings.c40_perception_dt)

        self.detection_strategy_type = tk.StringVar(value=PerceptionSettings.c12_detection_strategy)

        def _on_detection_change(*args):
            PerceptionSettings.c12_detection_strategy = self.detection_strategy_type.get()

        self.detection_strategy_type.trace_add("write", _on_detection_change)

        self.tracking_strategy_type = tk.StringVar(value=PerceptionSettings.c12_tracking_strategy)

        def _on_tracking_change(*args):
            PerceptionSettings.c12_tracking_strategy = self.tracking_strategy_type.get()

        self.tracking_strategy_type.trace_add("write", _on_tracking_change)

        self.prediction_strategy_type = tk.StringVar(value=PerceptionSettings.c12_prediction_strategy)

        def _on_prediction_change(*args):
            PerceptionSettings.c12_prediction_strategy = self.prediction_strategy_type.get()

        self.prediction_strategy_type.trace_add("write", _on_prediction_change)

        self.localization_type = tk.StringVar(value=ExecutionSettings.c40_localization)

        def _on_localization_change(*args):
            ExecutionSettings.c40_localization = self.localization_type.get()

        self.localization_type.trace_add("write", _on_localization_change)
        self.localization_dt = tk.DoubleVar(value=ExecutionSettings.c40_localization_dt)

        def _on_localization_dt_change(*args):
            ExecutionSettings.c40_localization_dt = float(self.localization_dt.get())

        self.localization_dt.trace_add("write", _on_localization_dt_change)

        self.mapping_type = tk.StringVar(value=ExecutionSettings.c40_mapping)

        def _on_mapping_change(*args):
            ExecutionSettings.c40_mapping = self.mapping_type.get()

        self.mapping_type.trace_add("write", _on_mapping_change)

        self.global_planner_type = tk.StringVar(
            value=ExecutionSettings.c40_global_planner
            if ExecutionSettings.c40_global_planner
            else (list(GlobalPlannerStrategy.registry.keys())[0] if GlobalPlannerStrategy.registry else None)
        )

        def _on_global_plan_change(*args):
            ExecutionSettings.c40_global_planner = self.global_planner_type.get()

        self.global_planner_type.trace_add("write", _on_global_plan_change)

        self.local_planner_type = tk.StringVar(
            value=(list(LocalPlanningStrategy.registry.keys())[0] if LocalPlanningStrategy.registry else None)
        )

        def _on_local_plan_change(*args):
            ExecutionSettings.c40_local_planner = self.local_planner_type.get()

        self.local_planner_type.trace_add("write", _on_local_plan_change)

        self.lap = tk.StringVar(value="0")
        self.current_wp = tk.StringVar(value="0")

        self.controller_type = tk.StringVar(
            value=(list(ControlStrategy.registry.keys())[0] if ControlStrategy.registry else None)
        )

        def _on_controller_change(*args):
            ExecutionSettings.c40_controller = self.controller_type.get()

        self.controller_type.trace_add("write", _on_controller_change)

        self.global_plan_view = tk.BooleanVar(value=False)
        self.local_plan_view = tk.BooleanVar(value=False)

        self.executer_type = tk.StringVar(
            value=(list(Executer.registry.keys())[0] if Executer.registry else None)
        )

        def _on_executer_change(*args):
            ExecutionSettings.c40_executer_type = self.executer_type.get()

        self.executer_type.trace_add("write", _on_executer_change)

        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)
        self.exec_localize = tk.BooleanVar(value=True)
        self.exec_running = False

        self.control_dt = tk.DoubleVar(value=0.01)

        def _on_control_dt_change(*args):
            ExecutionSettings.c40_control_dt = float(self.control_dt.get())
            _sync_exec_dt("control_dt", self.control_dt.get())

        self.control_dt.trace_add("write", _on_control_dt_change)

        self.replan_dt = tk.DoubleVar(value=0.5)

        def _on_replan_dt_change(*args):
            ExecutionSettings.c40_replan_dt = float(self.replan_dt.get())
            _sync_exec_dt("replan_dt", self.replan_dt.get())

        self.replan_dt.trace_add("write", _on_replan_dt_change)

        self.perception_dt = tk.DoubleVar(value=0.5)

        def _on_perception_dt_change(*args):
            ExecutionSettings.c40_perception_dt = float(self.perception_dt.get())
            _sync_exec_dt("perception_dt", self.perception_dt.get())

        self.perception_dt.trace_add("write", _on_perception_dt_change)

        self.sim_dt = tk.DoubleVar(value=0.01)

        def _on_sim_dt_change(*args):
            ExecutionSettings.c40_sim_dt = float(self.sim_dt.get())

        self.sim_dt.trace_add("write", _on_sim_dt_change)

        self.execution_bridge = tk.StringVar(value=ExecutionSettings.c40_bridge)

        def _on_execution_bridge_change(*args):
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

        self.bridge_provide_ground_truth_detection = tk.BooleanVar(value=False)

        def _on_gt_change(*args):
            ExecutionSettings.c41_provide_ground_truth = self.bridge_provide_ground_truth_detection.get()

        self.bridge_provide_ground_truth_detection.trace_add("write", _on_gt_change)

        self.bridge_provide_rgb_image = tk.BooleanVar(value=False)

        def _on_rgb_change(*args):
            ExecutionSettings.c41_provide_rgb = self.bridge_provide_rgb_image.get()

        self.bridge_provide_rgb_image.trace_add("write", _on_rgb_change)
        self.bridge_provide_depth_image = tk.BooleanVar(value=False)
        self.bridge_provide_lidar_data = tk.BooleanVar(value=False)

        def _on_lidar_change(*args):
            ExecutionSettings.c41_provide_lidar = self.bridge_provide_lidar_data.get()

        self.bridge_provide_lidar_data.trace_add("write", _on_lidar_change)

        self.log_level = tk.StringVar(value=ExecutionSettings.c40_log_level)

        def _on_log_level_change(*_):
            ExecutionSettings.c40_log_level = self.log_level.get()

        self.log_level.trace_add("write", _on_log_level_change)
        self.show_core_logs = tk.BooleanVar(value=True)
        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
        self.show_common_logs = tk.BooleanVar(value=True)
        self.show_plugins_logs = tk.BooleanVar(value=True)
        self.disable_log = tk.BooleanVar(value=False)
        self.max_log_lines = 1000
        self.log_view_expanded = tk.BooleanVar(value=False)
        self.log_view_default_height = tk.IntVar(value=12)
        self.log_view_expended_height = tk.IntVar(value=35)
        self.log_font = tk.StringVar(value="Courier")
        self.log_font_size = tk.IntVar(value=11)
        self.log_to_file = tk.BooleanVar(value=ExecutionSettings.c40_log_to_file)

        def _on_log_to_file_change(*_):
            ExecutionSettings.c40_log_to_file = self.log_to_file.get()

        self.log_to_file.trace_add("write", _on_log_to_file_change)
        self.log_pull_time = 50
        self.bg_color = "#333333" if self.dark_mode.get() else "white"
        self.fg_color = "white" if self.dark_mode.get() else "black"
        self.profile_list = []
