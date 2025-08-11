from __future__ import annotations
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk
import time
from c40_execution.c44_basic_sim import BasicSim
from c40_execution.c45_carla_bridge import CarlaBridge

from c40_execution.c49_settings import ExecutionSettings


if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

import logging

log = logging.getLogger(__name__)


class ExecView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)

        self.root = root

        # ----------------------------------------------------------------------
        ## Execute Frame
        # ----------------------------------------------------------------------
        self.execution_frame = ttk.LabelFrame(self, text="Execution")
        self.execution_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        exec_first_frame = ttk.Frame(self.execution_frame)
        exec_first_frame.pack(fill=tk.X)
        exec_second_frame = ttk.Frame(self.execution_frame)
        exec_second_frame.pack(fill=tk.X)
        exec_third_frame = ttk.Frame(self.execution_frame)
        exec_third_frame.pack(fill=tk.X)

        ttk.Label(exec_first_frame, text="Control Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        dt_control_entry = ttk.Entry(
            exec_first_frame,
            textvariable=self.root.setting.control_dt,
            width=5,
        )
        # self.dt_exec_cn_entry.insert(0, "0.02")
        dt_control_entry.pack(side=tk.LEFT)
        dt_control_entry.bind("<Return>", self.text_on_enter)

        ttk.Label(exec_first_frame, text="Replan Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        dt_plan_entry = ttk.Entry(
            exec_first_frame,
            textvariable=self.root.setting.replan_dt,
            width=5,
        )
        dt_plan_entry.pack(side=tk.LEFT)
        dt_plan_entry.bind("<Return>", self.text_on_enter)

        ttk.Label(exec_first_frame, text="Sim Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        sim_dt=ttk.Entry(
            exec_first_frame,
            textvariable=self.root.setting.sim_dt,
            width=5,
        )
        sim_dt.pack(side=tk.LEFT)
        sim_dt.bind("<Return>", self.text_on_enter)

        self.asyc_exec_cb = ttk.Checkbutton(
            exec_first_frame,
            text="Async Mode",
            command=self.__on_exec_change,
            variable=self.root.setting.async_exec,
        )
        self.asyc_exec_cb.pack(side=tk.RIGHT)

        self.start_exec_button = ttk.Button(
            exec_second_frame,
            text="Start",
            command=self.toggle_exec,
            style="Start.TButton",
            width=10,
        )
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT)

        ttk.Button(
            exec_second_frame,
            text="Stop",
            command=self.stop_exec,
            style="Stop.TButton",
        ).pack(side=tk.LEFT, padx=1)
        ttk.Button(exec_second_frame, text="Step", command=self.step_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Reset", command=self.reset_exec).pack(side=tk.LEFT)

        ttk.Label(exec_third_frame, text="Bridge:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            exec_third_frame,
            text="Basic Sim",
            variable=self.root.setting.execution_bridge,
            value=BasicSim.__name__,
            command=lambda: self.root.reload_stack(reload_code=False),
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            exec_third_frame,
            text="Carla",
            variable=self.root.setting.execution_bridge,
            value=CarlaBridge.__name__,
            command=lambda: self.root.reload_stack(reload_code=False),
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            exec_third_frame,
            text="Gazebo Ign",
            variable=self.root.setting.execution_bridge,
            value="GazeboIgnitionBridge",
            command=lambda: self.root.reload_stack(reload_code=False),
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(exec_third_frame, text="Control", variable=self.root.setting.exec_control).pack(side=tk.RIGHT)
        ttk.Checkbutton(exec_third_frame, text="Plan", variable=self.root.setting.exec_plan).pack(side=tk.RIGHT)
        ttk.Checkbutton(exec_third_frame, text="Percieve", variable=self.root.setting.exec_perceive).pack(side=tk.RIGHT)

        # ----------------------------------------------------------------------
        # Visualize frame setup
        # ----------------------------------------------------------------------
        self.visualize_frame = ttk.LabelFrame(self, text="Visualize")
        self.visualize_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ## UI Elements for Visualize - Checkboxes
        checkboxes_frame = ttk.Frame(self.visualize_frame)
        checkboxes_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Legend",
            variable=self.root.setting.show_legend,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Locations",
            variable=self.root.setting.show_past_locations,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Global Plan",
            variable=self.root.setting.show_global_plan,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Plan",
            variable=self.root.setting.show_local_plan,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Lattice",
            variable=self.root.setting.show_local_lattice,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="State",
            variable=self.root.setting.show_state,
            command=self.root.update_ui,
        ).pack(anchor=tk.W, side=tk.LEFT)

        ## UI Elements for Visualize - Buttons
        zoom_global_frame = ttk.Frame(self.visualize_frame)
        zoom_global_frame.pack(fill=tk.X, padx=5)
        ttk.Label(zoom_global_frame, text="Global 🔍").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(zoom_global_frame, text="➕", width=2, command=self.root.local_plan_plot_view.zoom_in).pack(
            side=tk.LEFT
        )
        ttk.Button(zoom_global_frame, text="➖", width=2, command=self.root.local_plan_plot_view.zoom_out).pack(
            side=tk.LEFT
        )
        ttk.Checkbutton(
            zoom_global_frame, text="Follow Planner", variable=self.root.setting.global_view_follow_planner
        ).pack(side=tk.LEFT)

        ttk.Label(zoom_global_frame, text="Real time: ", font=self.root.small_font).pack(
            anchor=tk.W, side=tk.LEFT, padx=5
        )
        ttk.Label(zoom_global_frame, textvariable=self.root.setting.elapsed_real_time, font=self.root.small_font).pack(
            anchor=tk.W, side=tk.LEFT, padx=10
        )

        ttk.Label(zoom_global_frame, textvariable=self.root.setting.replan_fps, font=self.root.small_font).pack(
            anchor=tk.W, side=tk.RIGHT, padx=5
        )

        ttk.Label(zoom_global_frame, text="Plan FPS: ", font=self.root.small_font).pack(anchor=tk.W, side=tk.RIGHT)

        zoom_frenet_frame = ttk.Frame(self.visualize_frame)
        zoom_frenet_frame.pack(fill=tk.X, padx=5)
        ttk.Label(zoom_frenet_frame, text="Frenet 🔍").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(
            zoom_frenet_frame,
            text="➕",
            width=2,
            command=self.root.local_plan_plot_view.zoom_in_frenet,
        ).pack(side=tk.LEFT)
        ttk.Button(
            zoom_frenet_frame,
            text="➖",
            width=2,
            command=self.root.local_plan_plot_view.zoom_out_frenet,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            zoom_frenet_frame, text="Follow Planner", variable=self.root.setting.frenet_view_follow_planner
        ).pack(side=tk.LEFT)

        ttk.Label(zoom_frenet_frame, text="Sim time : ", font=self.root.small_font).pack(
            anchor=tk.W, side=tk.LEFT, padx=5
        )
        ttk.Label(zoom_frenet_frame, textvariable=self.root.setting.elapsed_sim_time, font=self.root.small_font).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Label(zoom_frenet_frame, textvariable=self.root.setting.control_fps, font=self.root.small_font).pack(
            anchor=tk.W, side=tk.RIGHT
        )
        ttk.Label(zoom_frenet_frame, text="Con. FPS: ", font=self.root.small_font).pack(anchor=tk.W, side=tk.RIGHT)

    # --------------------------------------------------------------------------------------------
    # -SIM----------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def __on_exec_change(self):
        self.root.reload_stack(reload_code=False)

    def text_on_enter(self, event):
        widget = event.widget  # Get the widget that triggered the event
        text = widget.get()    # Retrieve the text from the widget
        self.root.validate_float_input(text)  # Validate the input
        log.debug("Text entered: %s", text)
        widget.tk_focusNext().focus_set()  # Move focus to the next widget

    def toggle_exec(self):
        if self.root.setting.exec_running:
            self.stop_exec()
            return
        self.root.setting.exec_running = True
        # self.start_exec_button.config(state=tk.DISABLED)
        self.start_exec_button.state(['disabled'])
        self.root.update_ui()
        self._exec_loop()

    def _exec_loop(self):
        if self.root.setting.exec_running:
            current_time = time.time()
            cn_dt = float(self.root.setting.control_dt.get())
            pl_dt = float(self.root.setting.replan_dt.get())
            sim_dt = float(self.root.setting.sim_dt.get())

            self.root.exec.step(
                control_dt=cn_dt,
                replan_dt=pl_dt,
                sim_dt=sim_dt,
                call_replan=self.root.setting.exec_plan.get(),
                call_control=self.root.setting.exec_control.get(),
                call_perceive=self.root.setting.exec_perceive.get(),
            ),
            self.root.update_ui()

            processing_time = time.time() - current_time
            next_frame_delay = max(0.001, sim_dt - processing_time)  # Ensure positive delay

            log.debug(f"Total Processing Time: {int(processing_time*1000):3d} ms")
            # self.root.after(int(sim_dt * 1000), self._exec_loop)
            self.root.after(int(next_frame_delay * 1000), self._exec_loop)

    def stop_exec(self):
        if self.root.setting.async_exec.get():
            log.info(f"Stopping Async Exec in 0.1 sec.")
            # self.root.after(100, self.root.exec.stop())
            self.root.exec.stop()
        # self.start_exec_button.config(state=tk.NORMAL)
        self.start_exec_button.state(['!disabled'])
        self.root.update_ui()
        self.root.setting.exec_running = False

    def step_exec(self):
        cn_dt = float(self.root.setting.control_dt.get())
        pl_dt = float(self.root.setting.replan_dt.get())
        self.root.exec.step(
            control_dt=cn_dt,
            replan_dt=pl_dt,
            call_replan=self.root.setting.exec_plan.get(),
            call_control=self.root.setting.exec_control.get(),
            call_perceive=self.root.setting.exec_perceive.get(),
        )
        self.root.update_ui()

    def reset_exec(self):
        self.root.exec.reset()
        self.root.update_ui()
