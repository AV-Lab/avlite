from __future__ import annotations
import tkinter as tk
import logging

from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c40_execution.c41_execution_model import Executer

log = logging.getLogger(__name__)

class VisualizationSettings:
    exclude = ["exclude","vehicle_state", "elapsed_real_time", "elapsed_sim_time", "lap", "replan_fps",
                         "control_fps", "current_wp", "exec_running", "profile_list", "perception_status_text", "extension_list"]
    filepath: str="configs/c50_visualization.yaml"

    def __init__(self):
        # Config
        self.shortcut_mode = tk.BooleanVar()
        self.dark_mode = tk.BooleanVar(value=True)
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

        self.xy_zoom = 30
        self.frenet_zoom = 30
        self.global_zoom = 30

        #############################
        # Perc Plan Control

        # perceptoin
        self.show_occupancy_flow = tk.BooleanVar(value=False)
        self.vehicle_state = tk.StringVar(value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.perception_status_text = tk.StringVar(value="Spawn Agent: Right click on the plot.")

        self.perception_type = tk.StringVar(value=list(PerceptionStrategy.registry.keys())[0] if PerceptionStrategy.registry else None)
        def _on_perception_change(*args):
            ExecutionSettings.perception = self.perception_type.get()
        self.perception_type.trace_add("write", _on_perception_change)
        self.perception_dt = tk.DoubleVar(value=ExecutionSettings.perception_dt)

        # planning
        self.global_planner_type = tk.StringVar(value=list(GlobalPlannerStrategy.registry.keys())[0] if GlobalPlannerStrategy.registry else None)
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

        self.exec_running = False # excluded

        self.control_dt = tk.DoubleVar(value=0.01)
        def _on_control_dt_change(*args):
            ExecutionSettings.control_dt = float(self.control_dt.get())
        self.control_dt.trace_add("write", _on_control_dt_change)
    
        self.replan_dt = tk.DoubleVar(value=0.5)
        def _on_replan_dt_change(*args):
            ExecutionSettings.replan_dt = float(self.replan_dt.get())
        self.replan_dt.trace_add("write", _on_replan_dt_change) 

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
        self.bridge_provide_rgb_image = tk.BooleanVar(value=False)  # Whether the world supports RGB image
        self.bridge_provide_depth_image = tk.BooleanVar(value=False)  # Whether the world supports depth image
        self.bridge_provide_lidar_data = tk.BooleanVar(value=False)  # Whether the world supports LiDAR data
        ############################


        ############################
        # APP Options

        # Logger Options
        self.log_level = tk.StringVar(value="INFO")
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

        self.log_to_file = tk.BooleanVar(value=False)

        self.log_pull_time = 50 # Time in milliseconds to pull logs from the logger
        
        # General variables - Not saved


        self.bg_color = "#333333" if self.dark_mode.get() else "white"
        self.fg_color = "white" if self.dark_mode.get() else "black"

        self.profile_list = []
        ############################

