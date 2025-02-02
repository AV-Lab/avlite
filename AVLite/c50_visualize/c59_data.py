from __future__ import annotations
import tkinter as tk

import logging
log = logging.getLogger(__name__)



class VisualizerData:
    def __init__(self, only_visualize: bool = False):
        self.shortcut_mode = tk.BooleanVar(value=only_visualize)
        self.dark_mode = tk.BooleanVar(value=True)

        # Plot options
        self.show_legend = tk.BooleanVar(value=False) # causes slow
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_global_plan = tk.BooleanVar(value=True)
        self.show_local_plan = tk.BooleanVar(value=True)
        self.show_local_lattice = tk.BooleanVar(value=True)
        self.show_state = tk.BooleanVar(value=True)
        self.global_view_follow_planner = tk.BooleanVar(value=False)
        self.frenet_view_follow_planner = tk.BooleanVar(value=False)

        # Exec Options
        self.async_exec = tk.BooleanVar(value=False)
        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)
        self.exec_running = False

        self.control_dt = tk.DoubleVar(value=0.01)
        self.replan_dt = tk.DoubleVar(value=0.5)

        # Logger Options
        self.exec_option = tk.StringVar(value="Basic")
        self.log_level = tk.StringVar(value="INFO")

        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
        

        self.xy_zoom = 30
        self.frenet_zoom = 30



