from __future__ import annotations
from typing import TYPE_CHECKING
from utils import save_visualizer_config, load_visualizer_config
if TYPE_CHECKING:
    from c50_visualize.c51_visualizer_app import VisualizerApp
import tkinter as tk
from tkinter import ttk


import logging

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c50_visualize.c51_visualizer_app import VisualizerApp


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
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.step_control())
        self.root.bind("g", lambda e: self.root.perceive_plan_control_view.align_control())

        self.root.bind("a", lambda e: self.root.perceive_plan_control_view.step_steer_left())
        self.root.bind("d", lambda e: self.root.perceive_plan_control_view.step_steer_right())
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
            variable=self.root.data.shortcut_mode,
            command=self.root.toggle_shortcut_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            self,
            text="Dark Mode",
            variable=self.root.data.dark_mode,
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
Plan:     n - Step plan        r - Replan            
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
        self.set_dark_mode() if self.root.data.dark_mode.get() else self.set_light_mode()

    def toggle_dark_mode_shortcut(self):
        if self.root.data.dark_mode.get():
            self.root.data.dark_mode.set(False)
        else:
            self.root.data.dark_mode.set(True)
        self.toggle_dark_mode()

    def set_dark_mode(self):
        self.root.configure(bg="black")
        self.root.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
        self.help_text.config(bg="gray14", fg="white", highlightbackground="black")
        self.root.local_plan_plot_view.set_plot_theme(bg_color="#2d2d2d", fg_color="white")
        log.info("Dark mode enabled.")

        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self.root)
            style.set_theme("equilux")
            # style.configure("TProgressbar",
            #                 thickness=10,
            #                 troughcolor='black',  # Background color
            #                 background='white',  # Pointer color
            #                 bordercolor='gray',  # Border color
            #                 lightcolor='white',  # Light part color
            #                 darkcolor='gray')    # Dark part color

        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")

    def set_light_mode(self):
        self.root.configure(bg="white")
        self.root.log_view.log_area.config(bg="white", fg="black")
        self.help_text.config(bg="white", fg="black")

        self.root.local_plan_plot_view.set_plot_theme(bg_color="white", fg_color="black")
        log.info("Light mode enabled.")
        # reset the theme
        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self.root)
            style.set_theme("yaru")
        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")

    def save_config(self):
        save_visualizer_config(self.root.data)
    def load_config(self):
        load_visualizer_config(self.root.data)
