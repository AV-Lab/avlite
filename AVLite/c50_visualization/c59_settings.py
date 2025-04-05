from __future__ import annotations
import tkinter as tk
from c20_planning.c21_base_global_planner import BaseGlobalPlanner
from c20_planning.c24_base_local_planner import BaseLocalPlanner


class VisualizationSettings:
    exclude = ["vehicle_state", "elapsed_real_time", "elapsed_sim_time", "lap", "replan_fps",
                         "control_fps", "current_wp", "exec_running"]
    filepath: str="configs/c50_visualization.yaml"

    def __init__(self, only_visualize: bool = False):
        self.shortcut_mode = tk.BooleanVar(value=only_visualize)
        self.dark_mode = tk.BooleanVar(value=True)

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

        # Perc Plan Control
        self.global_planner_type = tk.StringVar(
            value=list(BaseGlobalPlanner.registry.keys())[0]
                   if BaseGlobalPlanner.registry else None)

        self.enable_joystick = tk.BooleanVar(value=True)
        self.global_plan_view = tk.BooleanVar(value=False)
        self.local_plan_view = tk.BooleanVar(value=False)
        
        self.local_planner_type = tk.StringVar(
            value=(list(BaseLocalPlanner.registry.keys())[0] 
                   if BaseLocalPlanner.registry else None))

        # Exec Options
        self.async_exec = tk.BooleanVar(value=False)
        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)
        self.exec_running = False

        self.control_dt = tk.DoubleVar(value=0.01)
        self.replan_dt = tk.DoubleVar(value=0.5)
        self.sim_dt = tk.DoubleVar(value=0.01)

        # Logger Options
        self.execution_bridge = tk.StringVar(value="Basic")
        self.log_level = tk.StringVar(value="INFO")

        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
        self.disable_log = tk.BooleanVar(value=False)

        # General variables - Not saved
        self.replan_fps = tk.StringVar(value="0")
        self.control_fps = tk.StringVar(value="0")

        self.lap = tk.StringVar(value="0")
        self.elapsed_real_time = tk.StringVar(value="0")
        self.elapsed_sim_time = tk.StringVar(value="0")

        self.vehicle_state = tk.StringVar(
            value="Ego: (0.00, 0.00), Vel: 0.00 (0.00 km/h), θ: 0.0")
        self.current_wp = tk.StringVar(value="0")
