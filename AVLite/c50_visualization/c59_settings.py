from __future__ import annotations
import tkinter as tk
import logging

from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from c30_control.c32_control_strategy import ControlStrategy
from c40_execution.c49_settings import ExecutionSettings

log = logging.getLogger(__name__)

class VisualizationSettings:
    exclude = ["vehicle_state", "elapsed_real_time", "elapsed_sim_time", "lap", "replan_fps",
                         "control_fps", "current_wp", "exec_running", "profile_list", "perception_status_text"]
    filepath: str="configs/c50_visualization.yaml"

    def __init__(self):
        self.shortcut_mode = tk.BooleanVar()
        self.dark_mode = tk.BooleanVar(value=True)
        self.selected_profile = tk.StringVar(value="default")
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

        # Perc Plan Control
        self.global_planner_type = tk.StringVar(value=list(GlobalPlannerStrategy.registry.keys())[0] if GlobalPlannerStrategy.registry else None)

        self.enable_joystick = tk.BooleanVar(value=True)
        self.global_plan_view = tk.BooleanVar(value=False)
        self.local_plan_view = tk.BooleanVar(value=False)
        
        self.local_planner_type = tk.StringVar(value=(list(LocalPlannerStrategy.registry.keys())[0] if LocalPlannerStrategy.registry else None))
        self.controller_type = tk.StringVar(value=(list(ControlStrategy.registry.keys())[0] if ControlStrategy.registry else None))

        # Exec Options
        self.async_exec = tk.BooleanVar(value=False)
        def _on_async_exec_change(*args): # synchronize with ExecutionSettings
            ExecutionSettings.async_mode = bool(self.async_exec.get())
        self.async_exec.trace_add("write", _on_async_exec_change)
    
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

        self.execution_bridge = tk.StringVar(value="Basic")
        def _on_execution_bridge_change(*args):
            ExecutionSettings.bridge = self.execution_bridge.get()
        self.execution_bridge.trace_add("write", _on_execution_bridge_change)


        # Logger Options
        self.log_level = tk.StringVar(value="INFO")
        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
        self.show_tools_logs = tk.BooleanVar(value=True)
        self.disable_log = tk.BooleanVar(value=False)

        
        # General variables - Not saved
        self.replan_fps = tk.StringVar(value="0")
        self.control_fps = tk.StringVar(value="0")

        self.lap = tk.StringVar(value="0")
        self.elapsed_real_time = tk.StringVar(value="0")
        self.elapsed_sim_time = tk.StringVar(value="0")

        self.vehicle_state = tk.StringVar(
            value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        
        self.perception_status_text = tk.StringVar(value="Spawn Agent: Click on the plot.")

        self.current_wp = tk.StringVar(value="0")

        self.bg_color = "#333333" if self.dark_mode.get() else "white"
        self.fg_color = "white" if self.dark_mode.get() else "black"

        self.profile_list = []

