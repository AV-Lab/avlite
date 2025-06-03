from __future__ import annotations
from typing import TYPE_CHECKING
from c60_tools.c61_utils import save_config, load_setting
if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp
import tkinter as tk
from tkinter import ttk
from tkinter import TclError


import logging

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp


class ConfigShortcutView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Config")


        self.root: VisualizerApp = root
        # ----------------------------------------------------------------------
        # Key Bindings --------------------------------------------------------
        # ----------------------------------------------------------------------
        self.root.bind("Q", lambda e: self.root.quit())
        self.root.bind("R", lambda e: self.root.reload_stack())
        self.root.bind("D", lambda e: self.toggle_dark_mode_shortcut())
        self.root.bind("S", lambda e: self.root.toggle_shortcut_mode())

        self.root.bind("x", lambda e: self.root.exec_visualize_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.exec_visualize_view.step_exec())
        self.root.bind("t", lambda e: self.root.exec_visualize_view.reset_exec())

        self.root.bind("n", lambda e: self.root.perceive_plan_control_view.step_plan())
        self.root.bind("b", lambda e: self.root.perceive_plan_control_view.step_waypoint_back())
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.step_control())
        self.root.bind("g", lambda e: self.root.perceive_plan_control_view.align_control())

        self.root.bind("<KeyPress-a>", lambda e: self.root.perceive_plan_control_view.step_steer_left())
        self.root.bind("<KeyPress-d>", lambda e: self.root.perceive_plan_control_view.step_steer_right())
        self.root.bind("<KeyRelease-a>", lambda e: self.root.perceive_plan_control_view.reset_steer())
        self.root.bind("<KeyRelease-d>", lambda e: self.root.perceive_plan_control_view.reset_steer())
        self.root.bind("w", lambda e: self.root.perceive_plan_control_view.step_acc())
        self.root.bind("s", lambda e: self.root.perceive_plan_control_view.step_dec())

        self.root.bind("<Control-plus>", lambda e: self.root.local_plan_plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.local_plan_plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.local_plan_plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.local_plan_plot_view.zoom_out())

        ttk.Button(self, text="Reload Code", command=self.root.reload_stack).pack(side=tk.RIGHT)
        ttk.Button(self, text="Load Config", command=self.load_config).pack(side=tk.RIGHT)
        ttk.Button(self, text="Save Config", command=self.save_config).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            self,
            text="Shortcut Mode",
            variable=self.root.setting.shortcut_mode,
            command=self.root.toggle_shortcut_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            self,
            text="Dark Mode",
            variable=self.root.setting.dark_mode,
            command=self.toggle_dark_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
        # TODO: this is a dirty trick because its parent is root
        self.shortcut_frame = ttk.LabelFrame(root, text="Shortcuts")
        self.help_text = tk.Text(self.shortcut_frame, wrap=tk.WORD, width=50, height=7)
        key_binding_info = """
Perceive: 
Plan:     n - Step plan        b - Step Back                r - Replan            
Control:  h - Control Step     g - Re-align control         w - Accelerate 
          a - Steer left       d - Steer right              s - Deccelerate
Visalize: Q - Quit             S - Toggle shortcut          D - Toggle Dark Mode        R - Reload imports     
          + - Zoom In          - - Zoom Out           <Ctrl+> - Zoom In F         <Ctrl-> - Zoom Out F
Execute:  c - Step Execution   t - Reset execution          x - Toggle execution
         """.strip()
        self.help_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.help_text.insert(tk.END, key_binding_info)
        self.help_text.config(state=tk.DISABLED)  # Make the text area read-only


    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.dark_mode.get() else self.root.set_light_mode()

    def toggle_dark_mode_shortcut(self):
        # if self.root.setting.dark_mode.get():
        #     self.root.setting.dark_mode.set(False)
        # else:
        #     self.root.setting.dark_mode.set(True)

        self.toggle_dark_mode()


    def save_config(self):
        save_config(self.root.setting)
    def load_config(self):
        load_setting(self.root.setting)
