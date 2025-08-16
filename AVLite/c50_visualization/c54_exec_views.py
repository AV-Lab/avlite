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
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        self.execution_frame = ttk.LabelFrame(self, text="Execution")
        self.execution_frame.grid(row=0,column=0, sticky="nsew")


        ## Bridge 
        bridge_frame = BridgeFrame(self.root, self)
        bridge_frame.grid(row=0, column=1, sticky="nsew")
        
        ## Execution Settings Frame
        exec_stats_frame = ExecStatsFrame(self.root, self)
        exec_stats_frame.grid(row=0, column=2, sticky="nsew")

        self.columnconfigure(0, weight=2)  # execution_frame wider
        # self.columnconfigure(1, weight=1)  # exec_setting_frame
        # self.columnconfigure(2, weight=1)  # bridge_frame
        

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        exec_first_frame = ttk.Frame(self.execution_frame)
        exec_first_frame.grid(row=0, column=0, sticky="we")
        exec_second_frame = ttk.Frame(self.execution_frame)
        exec_second_frame.grid(row=1, column=0, sticky="we")
        exec_third_frame = ttk.Frame(self.execution_frame)
        exec_third_frame.grid(row=2, column=0, sticky="we")
        self.execution_frame.columnconfigure(0, weight=1)
        
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

class ExecStatsFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Execution Stats")
        self.root = root

        ttk.Label(self, text="Real time", font=self.root.small_font).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(self, textvariable=self.root.setting.elapsed_real_time, font=self.root.small_font).grid(row=0, column=1, sticky=tk.E)

        ttk.Label(self, text="Sim time", font=self.root.small_font).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(self, textvariable=self.root.setting.elapsed_sim_time, font=self.root.small_font).grid(row=1, column=1, sticky=tk.E)
        
        ttk.Label(self, text="Perc. FPS", font=self.root.small_font).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(self, textvariable=self.root.setting.perception_fps, font=self.root.small_font).grid(row=2, column=1, sticky=tk.E)

        ttk.Label(self, text="Plan FPS", font=self.root.small_font).grid(row=3, column=0, sticky=tk.W)
        ttk.Label(self, textvariable=self.root.setting.replan_fps, font=self.root.small_font).grid(row=3, column=1, sticky=tk.E)

        ttk.Label(self, text="Con. FPS", font=self.root.small_font).grid(row=4, column=0, sticky=tk.W)
        ttk.Label(self, textvariable=self.root.setting.control_fps, font=self.root.small_font).grid(row=4, column=1, sticky=tk.E)



class BridgeFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Bridge Setting")
        self.root = root

        self.chk_ground_truth = ttk.Checkbutton(self, text="Truth Detection", variable=self.root.setting.bridge_provide_ground_truth_detection)
        self.chk_ground_truth.grid(row=0, column=0, sticky="w")

        self.chk_rgb_image = ttk.Checkbutton(self, text="RGB Image", variable=self.root.setting.bridge_provide_rgb_image)
        self.chk_rgb_image.grid(row=1, column=0, sticky="w")

        # self.chk_depth_image = ttk.Checkbutton(self, text="Provide Depth Image", variable=self.root.setting.bridge_provide_depth_image)
        # self.chk_depth_image.grid(row=2, column=0, sticky="w")

        self.chk_lidar_data = ttk.Checkbutton(self, text="LiDAR Data", variable=self.root.setting.bridge_provide_lidar_data)
        self.chk_lidar_data.grid(row=3, column=0, sticky="w")

    def update_bridge_settings(self):
        pass


