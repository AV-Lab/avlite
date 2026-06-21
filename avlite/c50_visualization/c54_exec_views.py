from __future__ import annotations
from os import wait
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk
import time

from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_visualization.c58_ui_lib import attach_schema_tooltip
from avlite.c50_visualization.c59_settings import VisualizationSettings
from avlite.c60_common.c61_capabilities import WorldCapability


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
        self.execution_factory_frame = ttk.LabelFrame(self, text="Execution")
        self.execution_factory_frame.grid(row=0,column=0,pady=5, sticky="nsew")

        executer_frame = ExecSettingsFrame(self.root, self)
        executer_frame.grid(row=0, column=1, pady=5, sticky="nsew")

        ## Bridge 
        self.bridge_frame = BridgeFrame(self.root, self)
        self.bridge_frame.grid(row=0, column=2,pady=5, sticky="nsew")
        
        ## Execution Settings Frame
        exec_stats_frame = ExecStatsFrame(self.root, self)
        exec_stats_frame.grid(row=0, column=3,pady=5, sticky="nsew")

        self.columnconfigure(0, weight=2)  # execution_frame wider
        # self.columnconfigure(1, weight=1)  # exec_setting_frame
        # self.columnconfigure(2, weight=1)  # bridge_frame
        

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        exec_first_frame = ttk.Frame(self.execution_factory_frame)
        exec_first_frame.grid(row=0, column=0, sticky="we")
        exec_second_frame = ttk.Frame(self.execution_factory_frame)
        exec_second_frame.grid(row=1, column=0, sticky="we")
        exec_third_frame = ttk.Frame(self.execution_factory_frame)
        exec_third_frame.grid(row=2, column=0, sticky="we")
        self.execution_factory_frame.columnconfigure(0, weight=1)
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        lbl = ttk.Label(exec_first_frame, text="Perception \u0394t ")
        lbl.pack(side=tk.LEFT, padx=5, pady=5)
        dt_perception_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.perception_dt, width=5,)
        dt_perception_entry.pack(side=tk.LEFT)
        dt_perception_entry.bind("<Return>", self.text_on_enter)
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_perception_dt")
        attach_schema_tooltip(dt_perception_entry, ExecutionSettings, "c40_perception_dt")

        lbl = ttk.Label(exec_first_frame, text="Replan Δt ")
        lbl.pack(side=tk.LEFT, padx=5, pady=5)
        dt_plan_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.replan_dt, width=5,)
        dt_plan_entry.pack(side=tk.LEFT)
        dt_plan_entry.bind("<Return>", self.text_on_enter)
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_replan_dt")
        attach_schema_tooltip(dt_plan_entry, ExecutionSettings, "c40_replan_dt")

        lbl = ttk.Label(exec_first_frame, text="Control Δt ")
        lbl.pack(side=tk.LEFT, padx=5, pady=5)
        dt_control_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.control_dt, width=5,)
        dt_control_entry.pack(side=tk.LEFT)
        dt_control_entry.bind("<Return>", self.text_on_enter)
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_control_dt")
        attach_schema_tooltip(dt_control_entry, ExecutionSettings, "c40_control_dt")

        lbl = ttk.Label(exec_first_frame, text="Sim Δt ")
        lbl.pack(side=tk.LEFT, padx=5, pady=5)
        sim_dt = ttk.Entry(exec_first_frame, textvariable=self.root.setting.sim_dt, width=5,)
        sim_dt.pack(side=tk.LEFT)
        sim_dt.bind("<Return>", self.text_on_enter)
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_sim_dt")
        attach_schema_tooltip(sim_dt, ExecutionSettings, "c40_sim_dt")

        self.executer_dropdown_menu = ttk.Combobox(exec_first_frame, textvariable=self.root.setting.executer_type, state="readonly",)
        self.executer_dropdown_menu["values"] = list(Executer.registry.keys())
        self.executer_dropdown_menu.state(["readonly"])
        self.executer_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.executer_dropdown_menu.pack(side=tk.RIGHT)
        attach_schema_tooltip(self.executer_dropdown_menu, ExecutionSettings, "c40_executer_type")



        ## Second frame
        self.start_exec_button = ttk.Button( exec_second_frame, text="Start", command=self.toggle_exec, style="Start.TButton", width=10,)
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT)

        ttk.Button( exec_second_frame, text="Stop", command=self.stop_exec, style="Stop.TButton",).pack(side=tk.LEFT, padx=1)
        ttk.Button(exec_second_frame, text="Step", width=4, command=self.step_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Reset", width=4, command=self.reset_exec).pack(side=tk.LEFT)


        ## Third frame 
        # ttk.Label(exec_third_frame, text="World Bridge: ").pack(side=tk.LEFT)
        # ttk.Radiobutton( exec_third_frame, text="Basic Sim", variable=self.root.setting.execution_bridge, value=BasicSim.__name__,
        #     command=lambda: self.root.reload_stack(reload_code=False),
        # ).pack(side=tk.LEFT)
        # ttk.Radiobutton( exec_third_frame, text="Carla", variable=self.root.setting.execution_bridge, value=CarlaBridge.__name__,
        #     command=lambda: self.root.reload_stack(reload_code=False),
        # ).pack(side=tk.LEFT)
        # ttk.Radiobutton( exec_third_frame, text="Gazebo Ign", variable=self.root.setting.execution_bridge, value="GazeboIgnitionBridge",
        #     command=lambda: self.root.reload_stack(reload_code=False),
        # ).pack(side=tk.LEFT)
        vehicle_state_label = ttk.Label(exec_third_frame, font=self.root.small_font, textvariable=self.root.setting.vehicle_state)
        vehicle_state_label.pack(side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)


        global_tj_file = ttk.Entry(exec_second_frame, textvariable=self.root.setting.default_global_plan_file, width=15,)
        global_tj_file.pack(side=tk.RIGHT, padx=5, pady=5)
        global_tj_file.bind("<Return>", self.text_on_enter)
        lbl = ttk.Label(exec_second_frame, text="Default Global Plan")
        lbl.pack(side=tk.RIGHT, padx=5, pady=5)
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_global_trajectory")
        attach_schema_tooltip(global_tj_file, ExecutionSettings, "c40_global_trajectory")


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
            pr_dt = float(self.root.setting.perception_dt.get())
            sim_dt = float(self.root.setting.sim_dt.get())

            self.root.exec.step(
                control_dt=cn_dt,
                replan_dt=pl_dt,
                perception_dt=pr_dt,
                sim_dt=sim_dt,
                call_replan=self.root.setting.exec_plan.get(),
                call_control=self.root.setting.exec_control.get(),
                call_perceive=self.root.setting.exec_perceive.get(),
                call_localize=self.root.setting.exec_localize.get(),
            ),

            # Throttle UI updates to 20 Hz regardless of step() speed.
            # This decouples simulation rate from widget redraw rate.
            _now = time.time()
            if _now - getattr(self, '_last_ui_update_time', 0) >= 0.05:
                self._last_ui_update_time = _now
                self.root.update_ui()

            processing_time = time.time() - current_time
            log.debug("Total Processing Time: %d ms", int(processing_time * 1000))
            # Ask the executer how fast the UI should poll it.
            # Executers with background workers return a fixed delay; others return None
            # to indicate the UI should derive the delay from sim_dt adaptively.
            _poll_delay = self.root.exec.ui_poll_delay
            if _poll_delay is not None:
                next_frame_delay = _poll_delay
            else:
                next_frame_delay = max(0.001, sim_dt - processing_time)
            self.root.after(int(next_frame_delay * 1000), self._exec_loop)

    def stop_exec(self):
        self.root.exec.stop()
        # self.start_exec_button.config(state=tk.NORMAL)
        self.start_exec_button.state(['!disabled'])
        self.root.update_ui()
        self.root.setting.exec_running = False

    def step_exec(self):
        cn_dt = float(self.root.setting.control_dt.get())
        pl_dt = float(self.root.setting.replan_dt.get())
        pr_dt = float(self.root.setting.perception_dt.get())
        self.root.exec.step(
            control_dt=cn_dt,
            replan_dt=pl_dt,
            perception_dt=pr_dt,
            call_replan=self.root.setting.exec_plan.get(),
            call_control=self.root.setting.exec_control.get(),
            call_perceive=self.root.setting.exec_perceive.get(),
            call_localize=self.root.setting.exec_localize.get(),
        )
        self.root.update_ui()

    def update_data(self):
        """Refresh the executer dropdown from the registry."""
        self.executer_dropdown_menu["values"] = list(Executer.registry.keys())

    def reset_exec(self):
        self.root.exec.reset()
        self.root.update_ui()

class ExecSettingsFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Executables")
        self.root = root
        chk = ttk.Checkbutton(self, text="Control", variable=self.root.setting.exec_control)
        chk.grid(row=0, column=0, sticky="w")
        attach_schema_tooltip(chk, VisualizationSettings, "exec_control")
        chk = ttk.Checkbutton(self, text="Planning", variable=self.root.setting.exec_plan)
        chk.grid(row=1, column=0, sticky="w")
        attach_schema_tooltip(chk, VisualizationSettings, "exec_plan")
        chk = ttk.Checkbutton(self, text="Perception", variable=self.root.setting.exec_perceive)
        chk.grid(row=2, column=0, sticky="w")
        attach_schema_tooltip(chk, VisualizationSettings, "exec_perceive")
        chk = ttk.Checkbutton(self, text="Localization", variable=self.root.setting.exec_localize)
        chk.grid(row=3, column=0, sticky="w")
        attach_schema_tooltip(chk, VisualizationSettings, "exec_localize")


class BridgeFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Bridge Setting")
        self.root = root
        world_bridge_dropdown_menu = ttk.Combobox(self, textvariable=self.root.setting.execution_bridge, width=10, state="readonly",)
        world_bridge_dropdown_menu["values"] = list(WorldBridge.registry.keys())
        world_bridge_dropdown_menu.state(["readonly"])
        world_bridge_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        world_bridge_dropdown_menu.grid(row=0, column=0, pady=0, sticky="we")
        attach_schema_tooltip(world_bridge_dropdown_menu, ExecutionSettings, "c40_bridge")

        self.chk_ground_truth = ttk.Checkbutton(self, text="Ground Truth", variable=self.root.setting.bridge_provide_ground_truth_detection)
        self.chk_ground_truth.grid(row=1, column=0, sticky="w")
        attach_schema_tooltip(self.chk_ground_truth, ExecutionSettings, "c41_provide_ground_truth")

        self.chk_rgb_image = ttk.Checkbutton(self, text="RGB Image", variable=self.root.setting.bridge_provide_rgb_image)
        self.chk_rgb_image.grid(row=2, column=0, sticky="w")
        attach_schema_tooltip(self.chk_rgb_image, ExecutionSettings, "c41_provide_rgb")

        # self.chk_depth_image = ttk.Checkbutton(self, text="Provide Depth Image", variable=self.root.setting.bridge_provide_depth_image)
        # self.chk_depth_image.grid(row=2, column=0, sticky="w")

        self.chk_lidar_data = ttk.Checkbutton(self, text="LiDAR Data", variable=self.root.setting.bridge_provide_lidar_data)
        self.chk_lidar_data.grid(row=3, column=0, sticky="w")
        attach_schema_tooltip(self.chk_lidar_data, ExecutionSettings, "c41_provide_lidar")



    def update_for_bridge(self, capabilities: set):
        """Enable / disable checkboxes based on the active bridge's capabilities."""
        cap_map = {
            WorldCapability.GT_DETECTION: self.chk_ground_truth,
            WorldCapability.CAMERA_RGB:    self.chk_rgb_image,
        }
        for cap, chk in cap_map.items():
            if cap in capabilities:
                chk.state(['!disabled'])
            else:
                chk.state(['disabled'])

        # LiDAR checkbox: enabled when the bridge provides either 2D or 3D LiDAR
        if {WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D} & capabilities:
            self.chk_lidar_data.state(['!disabled'])
        else:
            self.chk_lidar_data.state(['disabled'])




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



