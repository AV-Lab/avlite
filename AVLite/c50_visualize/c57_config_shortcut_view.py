from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import c50_visualize.c52_plotlib as c52_plotlib

import logging
log = logging.getLogger(__name__)


class ConfigShortcutView():
    def __init__(self, root: VisualizerApp):
        self.root = root
        # ----------------------------------------------------------------------
        # Key Bindings --------------------------------------------------------
        # ----------------------------------------------------------------------
        self.root.bind("Q", lambda e: self.root.quit())
        self.root.bind("R", lambda e: self.reload_stack())
        self.root.bind("D", lambda e: self.toggle_dark_mode_shortcut())
        self.root.bind("S", lambda e: self.set_shortcut_mode())

        self.root.bind("x", lambda e: self.root.visualize_exec_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.visualize_exec_view.step_exec())
        self.root.bind("t", lambda e: self.root.visualize_exec_view.reset_exec())

        self.root.bind("n", lambda e: self.root.perceive_plan_control_view.step_plan())
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.step_control())
        self.root.bind("g", lambda e: self.root.perceive_plan_control_view.align_control())

        self.root.bind("a", lambda e: self.root.perceive_plan_control_view.step_steer_left())
        self.root.bind("d", lambda e: self.root.perceive_plan_control_view.step_steer_right())
        self.root.bind("w", lambda e: self.root.perceive_plan_control_view.step_acc())
        self.root.bind("s", lambda e: self.root.perceive_plan_control_view.step_dec())

        self.root.bind("<Control-plus>", lambda e: self.root.plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.plot_view.zoom_out())
        
        # ----------------------------------------------------------------------
        # Config frame
        # ------------------------------------------------------
        config_frame = ttk.LabelFrame(root, text="Config")
        config_frame.pack(fill=tk.X, side=tk.TOP)
        ttk.Button(config_frame, text="Reload Code", command=self.reload_stack).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            config_frame,
            text="Shortcut Mode",
        
            variable=self.root.data.shortcut_mode,
            command=self.update_UI,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            config_frame,
            text="Dark Mode",
            variable=self.root.data.dark_mode,
            command=self.toggle_dark_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
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
    

    def set_shortcut_mode(self):
        if self.root.data.shortcut_mode.get():
            self.root.data.shortcut_mode.set(False)
        else:
            self.root.data.shortcut_mode.set(True)
        self.update_UI()

    def update_UI(self):

        if self.root.data.shortcut_mode.get():
            self.root.visualize_exec_view.vis_exec_frame.pack_forget()
            self.root.perceive_plan_control_view.perceive_plan_control_frame.pack_forget()
            self.root.log_view.log_frame.pack_forget()

            self.shortcut_frame.pack(fill=tk.X, side=tk.TOP)
            self.root.log_view.log_frame.pack(fill=tk.X)
        else:
            self.shortcut_frame.pack_forget()
            self.root.log_view.log_frame.pack_forget()

            self.root.visualize_exec_view.vis_exec_frame.pack(fill=tk.X, side=tk.TOP)
            self.root.perceive_plan_control_view.perceive_plan_control_frame.pack(fill=tk.X)
            self.root.log_view.log_area.pack(fill=tk.BOTH, expand=True)
            self.root.log_view.log_frame.pack(fill=tk.X)

        # max_height = int(self.winfo_height() * 0.4)
        # self.log_frame.config(height=max_height)
        self.root.plot_view.replot()

    def toggle_dark_mode(self):
        self.set_dark_mode() if self.root.data.dark_mode.get() else self.light_mode()

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
        self.root.plot_view.set_plot_theme(bg_color="#2d2d2d", fg_color="white")

        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self.root)
            style.set_theme("equilux")

        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")

    def light_mode(self):
        self.root.configure(bg="white")
        self.root.log_view.log_area.config(bg="white", fg="black")
        self.help_text.config(bg="white", fg="black")
        self.root.plot_view.set_plot_theme(bg_color="white", fg_color="black")
        # reset the theme
        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self.root)
            style.set_theme("yaru")
        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")


    def reload_stack(self):
        if self.root.code_reload_function is not None:
            self.root.exec = self.root.code_reload_function()
            self.root.plot_view.replot()
        else:
            log.warning("No code reload function provided.")


