from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import time

import logging
log = logging.getLogger(__name__)

class VisualizeExecView:
    def __init__(self, root: VisualizerApp):
        self.root = root
        self.vis_exec_frame = ttk.Frame(root)
        self.vis_exec_frame.pack(fill=tk.X)

        # ----------------------------------------------------------------------
        ## Execute Frame
        # ----------------------------------------------------------------------
        self.execution_frame = ttk.LabelFrame(self.vis_exec_frame, text="Execute (Auto)")
        self.execution_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        exec_first_frame = ttk.Frame(self.execution_frame)
        exec_first_frame.pack(fill=tk.X)
        exec_second_frame = ttk.Frame(self.execution_frame)
        exec_second_frame.pack(fill=tk.X)
        exec_third_frame = ttk.Frame(self.execution_frame)
        exec_third_frame.pack(fill=tk.X)

        ttk.Label(exec_first_frame, text="Control Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_cn_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_cn_entry.insert(0, "0.02")
        self.dt_exec_cn_entry.pack(side=tk.LEFT)

        ttk.Label(exec_first_frame, text="Replan Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_pl_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_pl_entry.insert(0, "1.7")
        self.dt_exec_pl_entry.pack(side=tk.LEFT)

        gruvbox_green = "#b8bb26"
        gruvbox_light_green = "#fe8019"
        gruvbox_orange = "#d65d0e"
        self.start_exec_button = tk.Button(
            exec_second_frame,
            text="Start",
            command=self.toggle_exec,
            bg=gruvbox_orange,
            fg="white",
            borderwidth=0,
            highlightthickness=0,
            width=10,
        )
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT)

        ttk.Button(exec_second_frame, text="Stop", command=self.stop_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Step", command=self.step_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Reset", command=self.reset_exec).pack(side=tk.LEFT)

        ttk.Label(exec_third_frame, text="Bridge:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            exec_third_frame,
            text="Basic",
            variable=self.root.data.exec_option,
            value="Basic",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(exec_third_frame, text="ROS", variable=self.root.data.exec_option, value="ROS").pack(side=tk.LEFT)
        ttk.Radiobutton(exec_third_frame, text="Carla", variable=self.root.data.exec_option, value="Carla").pack(side=tk.LEFT)

        # ttk.Checkbutton(exec_third_frame, text="Control", variable=self.exec_control).pack(side=tk.RIGHT)
        # ttk.Checkbutton(exec_third_frame, text="Plan", variable=self.exec_plan).pack(side=tk.RIGHT)
        # ttk.Checkbutton(exec_third_frame, text="Percieve", variable=self.exec_perceive).pack(side=tk.RIGHT)

        # ----------------------------------------------------------------------
        # Visualize frame setup
        # ----------------------------------------------------------------------
        self.visualize_frame = ttk.LabelFrame(self.vis_exec_frame, text="Visualize")
        self.visualize_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ## UI Elements for Visualize - Checkboxes
        checkboxes_frame = ttk.Frame(self.visualize_frame)
        checkboxes_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Legend",
            variable=self.root.data.show_legend,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Locations",
            variable=self.root.data.show_past_locations,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Global Plan",
            variable=self.root.data.show_global_plan,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Plan",
            variable=self.root.data.show_local_plan,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Lattice",
            variable=self.root.data.show_local_lattice,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="State",
            variable=self.root.data.show_state,
            command=self.root.plot_view.replot,
        ).pack(anchor=tk.W, side=tk.LEFT)

        ## UI Elements for Visualize - Buttons
        zoom_global_frame = ttk.Frame(self.visualize_frame)
        zoom_global_frame.pack(fill=tk.X, padx=5)

        ttk.Label(zoom_global_frame, text="Global Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(zoom_global_frame, text="Zoom In", command=self.root.plot_view.zoom_in).pack(side=tk.LEFT)
        ttk.Button(zoom_global_frame, text="Zoom Out", command=self.root.plot_view.zoom_out).pack(side=tk.LEFT)
        zoom_frenet_frame = ttk.Frame(self.visualize_frame)
        zoom_frenet_frame.pack(fill=tk.X, padx=5)
        ttk.Label(zoom_frenet_frame, text="Frenet Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(zoom_frenet_frame, text="Zoom In", command=self.root.plot_view.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(zoom_frenet_frame, text="Zoom Out", command=self.root.plot_view.zoom_out_frenet).pack(side=tk.LEFT)

    # --------------------------------------------------------------------------------------------
    # -SIM----------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def toggle_exec(self):
        if self.root.data.animation_running:
            self.stop_exec()
            return
        self.root.data.animation_running = True
        self.start_exec_button.config(state=tk.DISABLED)
        self._exec_loop()

    def _exec_loop(self):
        if self.root.data.animation_running:
            cn_dt = float(self.dt_exec_cn_entry.get())
            pl_dt = float(self.dt_exec_pl_entry.get())

            self.root.exec.step(control_dt=cn_dt, replan_dt=pl_dt)
            self.root.plot_view.replot()
            self.root.perceive_plan_control_view.global_tj_wp_entry.delete(0, tk.END)
            self.root.perceive_plan_control_view.global_tj_wp_entry.insert(0, str(self.root.exec.planner.global_trajectory.next_wp - 1))
            self.root.after(int(cn_dt * 1000), self._exec_loop)

    def stop_exec(self):
        self.root.data.animation_running = False
        self.start_exec_button.config(state=tk.NORMAL)

    def step_exec(self):
        cn_dt = float(self.dt_exec_cn_entry.get())
        pl_dt = float(self.dt_exec_pl_entry.get())
        self.root.exec.step(control_dt=cn_dt, replan_dt=pl_dt)
        self.root.plot_view.replot()

    def reset_exec(self):
        self.root.exec.reset()
        self.root.plot_view.replot()
