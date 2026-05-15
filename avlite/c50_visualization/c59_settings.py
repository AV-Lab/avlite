from __future__ import annotations
import tkinter as tk
import logging

from avlite.c10_perception.c12_perception_strategy import (
    PerceptionStrategy, DetectionStrategy, TrackingStrategy, PredictionStrategy,
)
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c40_execution.c41_execution_model import Executer

log = logging.getLogger(__name__)


class VisualizationSettings:
    exclude = ["exclude","vehicle_state", "elapsed_real_time", "elapsed_sim_time", "lap", "replan_fps",
                         "control_fps", "perception_fps", "current_wp", "exec_running", "profile_list", "perception_status_text", "extension_list"]
    filepath: str="configs/c50_visualization.yaml"

    def __init__(self):
        # Config
        self.shortcut_mode = tk.BooleanVar()
        self.dark_mode = tk.BooleanVar(value=True)
        self.hide_menubar = tk.BooleanVar(value=False)
        self.selected_profile = tk.StringVar(value="default")
        self.next_profile = tk.StringVar(value="default")
        self.load_extensions = tk.BooleanVar(value=True)  # Load extensions on startup
        self.mouse_drag_slowdown_factor = 0.5

        # Plot options
        self.show_legend = tk.BooleanVar(value=False)  # causes slow
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_global_plan = tk.BooleanVar(value=True)
        self.show_local_plan = tk.BooleanVar(value=True)
        self.show_local_lattice = tk.BooleanVar(value=True)
        self.show_state = tk.BooleanVar(value=True)
        self.global_view_follow_planner = tk.BooleanVar(value=False)
        self.frenet_view_follow_planner = tk.BooleanVar(value=False)
        self.show_local_global_view = tk.BooleanVar(value=True)
        self.show_local_frenet_view = tk.BooleanVar(value=True)

        self.xy_zoom = 30
        self.frenet_zoom = 30
        self.global_zoom = 30

        #############################
        # Perc Plan Control

        # perception
        self.show_occupancy_flow = tk.BooleanVar(value=False)
        self.show_perception_extras = tk.BooleanVar(value=False)
        self.vehicle_state = tk.StringVar(value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.perception_status_text = tk.StringVar(value="Spawn Agent: Right click on the plot.")

        self.perception_type = tk.StringVar(value=list(PerceptionStrategy.registry.keys())[0] if PerceptionStrategy.registry else None)
        def _on_perception_change(*args):
            ExecutionSettings.perception = self.perception_type.get()
        self.perception_type.trace_add("write", _on_perception_change)
        self.perception_dt = tk.DoubleVar(value=ExecutionSettings.perception_dt)

        self.detection_strategy_type = tk.StringVar(value=PerceptionSettings.detection_strategy or "Ground Truth")
        def _on_detection_change(*args):
            v = self.detection_strategy_type.get()
            PerceptionSettings.detection_strategy = "" if v == "Ground Truth" else v
        self.detection_strategy_type.trace_add("write", _on_detection_change)

        self.tracking_strategy_type = tk.StringVar(value=PerceptionSettings.tracking_strategy or "Ground Truth")
        def _on_tracking_change(*args):
            v = self.tracking_strategy_type.get()
            PerceptionSettings.tracking_strategy = "" if v == "Ground Truth" else v
        self.tracking_strategy_type.trace_add("write", _on_tracking_change)

        self.prediction_strategy_type = tk.StringVar(value=PerceptionSettings.prediction_strategy or "Ground Truth")
        def _on_prediction_change(*args):
            v = self.prediction_strategy_type.get()
            PerceptionSettings.prediction_strategy = "" if v == "Ground Truth" else v
        self.prediction_strategy_type.trace_add("write", _on_prediction_change)

        # localization
        self.localization_type = tk.StringVar(value=list(LocalizationStrategy.registry.keys())[0] if LocalizationStrategy.registry else "Ground Truth")
        def _on_localization_change(*args):
            v = self.localization_type.get()
            ExecutionSettings.localization = "" if v == "Ground Truth" else v
        self.localization_type.trace_add("write", _on_localization_change)
        self.localization_dt = tk.DoubleVar(value=ExecutionSettings.localization_dt)
        def _on_localization_dt_change(*args):
            ExecutionSettings.localization_dt = float(self.localization_dt.get())
        self.localization_dt.trace_add("write", _on_localization_dt_change)

        # mapping
        self.mapping_type = tk.StringVar(value=list(MappingStrategy.registry.keys())[0] if MappingStrategy.registry else "Ground Truth")
        def _on_mapping_change(*args):
            v = self.mapping_type.get()
            ExecutionSettings.mapping = "" if v == "Ground Truth" else v
        self.mapping_type.trace_add("write", _on_mapping_change)

        # planning
        self.global_planner_type = tk.StringVar(value=ExecutionSettings.global_planner if ExecutionSettings.global_planner else (list(GlobalPlannerStrategy.registry.keys())[0] if GlobalPlannerStrategy.registry else None))
        def _on_global_plan_change(*args):
            ExecutionSettings.global_planner = self.global_planner_type.get()
        self.global_planner_type.trace_add("write", _on_global_plan_change)

        self.local_planner_type = tk.StringVar(value=(list(LocalPlannerStrategy.registry.keys())[0] if LocalPlannerStrategy.registry else None))
        def _on_local_plan_change(*args):
            ExecutionSettings.local_planner = self.local_planner_type.get()
        self.local_planner_type.trace_add("write", _on_local_plan_change)
        
        self.lap = tk.StringVar(value="0")
        self.current_wp = tk.StringVar(value="0")



        # control    
        self.controller_type = tk.StringVar(value=(list(ControlStrategy.registry.keys())[0] if ControlStrategy.registry else None))
        def _on_controller_change(*args):
            ExecutionSettings.controller = self.controller_type.get()
        self.controller_type.trace_add("write", _on_controller_change)


        self.enable_joystick = tk.BooleanVar(value=True)
        self.global_plan_view = tk.BooleanVar(value=False)
        self.local_plan_view = tk.BooleanVar(value=False)
        
        ############################

        ############################
        # Exec Options
        
        self.executer_type = tk.StringVar(value=(list(Executer.registry.keys())[0] if Executer.registry else None))
        def _on_executer_change(*args):
            ExecutionSettings.executer_type = self.executer_type.get()
        self.executer_type.trace_add("write", _on_executer_change)
    
        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)
        self.exec_localize = tk.BooleanVar(value=True)

        self.exec_running = False # excluded

        self.control_dt = tk.DoubleVar(value=0.01)
        def _on_control_dt_change(*args):
            ExecutionSettings.control_dt = float(self.control_dt.get())
            _sync_exec_dt("control_dt", self.control_dt.get())
        self.control_dt.trace_add("write", _on_control_dt_change)
    
        self.replan_dt = tk.DoubleVar(value=0.5)
        def _on_replan_dt_change(*args):
            ExecutionSettings.replan_dt = float(self.replan_dt.get())
            _sync_exec_dt("replan_dt", self.replan_dt.get())
        self.replan_dt.trace_add("write", _on_replan_dt_change)

        self.perception_dt = tk.DoubleVar(value=0.5)
        def _on_perception_dt_change(*args):
            ExecutionSettings.perception_dt = float(self.perception_dt.get())
            _sync_exec_dt("perception_dt", self.perception_dt.get())
        self.perception_dt.trace_add("write", _on_perception_dt_change)

        self.sim_dt = tk.DoubleVar(value=0.01)
        def _on_sim_dt_change(*args):
            ExecutionSettings.sim_dt = float(self.sim_dt.get())
        self.sim_dt.trace_add("write", _on_sim_dt_change)

        self.execution_bridge = tk.StringVar(value=ExecutionSettings.bridge)
        def _on_execution_bridge_change(*args):
            ExecutionSettings.bridge = self.execution_bridge.get()
        self.execution_bridge.trace_add("write", _on_execution_bridge_change)
        

        self.default_global_plan_file = tk.StringVar(value=ExecutionSettings.global_trajectory)
        def _on_default_global_plan_file_change(*args):
            ExecutionSettings.bridge = self.default_global_plan_file.get()
        self.default_global_plan_file.trace_add("write", _on_default_global_plan_file_change)
        
        self.elapsed_real_time = tk.StringVar(value="0")
        self.elapsed_sim_time = tk.StringVar(value="0")
        
        self.replan_fps = tk.StringVar(value="0")
        self.control_fps = tk.StringVar(value="0")
        self.perception_fps = tk.StringVar(value="0")


        ## World Bridge model
        self.bridge_provide_ground_truth_detection = tk.BooleanVar(value=False)  # Whether the world supports ground truth perception
        def _on_gt_change(*args):
            ExecutionSettings.provide_ground_truth = self.bridge_provide_ground_truth_detection.get()
        self.bridge_provide_ground_truth_detection.trace_add("write", _on_gt_change)

        self.bridge_provide_rgb_image = tk.BooleanVar(value=False)  # Whether the world supports RGB image
        def _on_rgb_change(*args):
            ExecutionSettings.provide_rgb = self.bridge_provide_rgb_image.get()
        self.bridge_provide_rgb_image.trace_add("write", _on_rgb_change)
        self.bridge_provide_depth_image = tk.BooleanVar(value=False)  # Whether the world supports depth image
        self.bridge_provide_lidar_data = tk.BooleanVar(value=False)  # Whether the world supports LiDAR data
        def _on_lidar_change(*args):
            ExecutionSettings.provide_lidar = self.bridge_provide_lidar_data.get()
        self.bridge_provide_lidar_data.trace_add("write", _on_lidar_change)
        ############################


        ############################
        # APP Options

        # Logger Options
        self.log_level = tk.StringVar(value=ExecutionSettings.log_level)
        def _on_log_level_change(*_):
            ExecutionSettings.log_level = self.log_level.get()
        self.log_level.trace_add("write", _on_log_level_change)
        self.show_core_logs = tk.BooleanVar(value=True)
        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
        self.show_common_logs = tk.BooleanVar(value=True)
        self.show_extensions_logs = tk.BooleanVar(value=True)
        self.disable_log = tk.BooleanVar(value=False)
        
        self.max_log_lines = 1000  # Maximum number of log lines to keep
        self.log_view_expanded = tk.BooleanVar(value=False)  # Whether the log view is expanded
        self.log_view_default_height = tk.IntVar(value=12)  # Height of the log view in lines
        self.log_view_expended_height = tk.IntVar(value=35) # Height of the log view when expanded in lines 
        self.log_font = tk.StringVar(value="Courier")  # Font for the log view
        self.log_font_size = tk.IntVar(value=11)

        self.log_to_file = tk.BooleanVar(value=ExecutionSettings.log_to_file)
        def _on_log_to_file_change(*_):
            ExecutionSettings.log_to_file = self.log_to_file.get()
        self.log_to_file.trace_add("write", _on_log_to_file_change)

        self.log_pull_time = 50 # Time in milliseconds to pull logs from the logger
        
        # General variables - Not saved


        self.bg_color = "#333333" if self.dark_mode.get() else "white"
        self.fg_color = "white" if self.dark_mode.get() else "black"

        self.profile_list = []
        ############################

    _GT_SENTINEL_VARS = (
        "detection_strategy_type",
        "tracking_strategy_type",
        "prediction_strategy_type",
        "localization_type",
        "mapping_type",
    )

    def normalize_gt_sentinels(self):
        """Convert empty-string values to 'Ground Truth' display label after profile load."""
        for name in self._GT_SENTINEL_VARS:
            var = getattr(self, name, None)
            if var is not None and isinstance(var, tk.StringVar) and var.get() == "":
                var.set("Ground Truth")


def _sync_exec_dt(attr: str, value: float) -> None:
    """Persist dt change to the ROS extension YAML so it takes effect on next launch."""
    try:
        from avlite.extensions.e40_executer_ROS2.settings import ExtensionSettings as ROSSettings
        from avlite.c60_common.c61_setting_utils import save_setting
        setattr(ROSSettings, attr, float(value))
        save_setting(ROSSettings)
    except Exception:
        pass

