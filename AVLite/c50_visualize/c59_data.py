from __future__ import annotations
import tkinter as tk

import logging
log = logging.getLogger(__name__)


class VisualizerData:
    def __init__(self, only_visualize: bool = False):
        # ----------------------------------------------------------------------
        # Variables for checkboxes --------------------------------------------
        # ----------------------------------------------------------------------
        self.shortcut_mode = tk.BooleanVar(value=only_visualize)
        self.dark_mode = tk.BooleanVar(value=True)

        self.show_legend = tk.BooleanVar(value=True)
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_global_plan = tk.BooleanVar(value=True)
        self.show_local_plan = tk.BooleanVar(value=True)
        self.show_local_lattice = tk.BooleanVar(value=True)
        self.show_state = tk.BooleanVar(value=True)

        # Exec Options
        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)

        self.animation_running = False

        self.exec_option = tk.StringVar(value="Basic")
        self.debug_option = tk.StringVar(value="INFO")

        # self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)
