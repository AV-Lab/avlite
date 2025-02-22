from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import time
from typing import TYPE_CHECKING
from c30_control.c31_base_controller import ControlComand
from c50_visualize.c58_ui_lib import ValueGauge
import logging

if TYPE_CHECKING:
    from c50_visualize.c51_visualizer_app import VisualizerApp

log = logging.getLogger(__name__)


class PerceivePlanControlView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root
        # ----------------------------------------------------------------------
        ## Perceive Frame
        # ----------------------------------------------------------------------
        self.perceive_frame = ttk.LabelFrame(self, text="Perceive")
        self.perceive_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Progress bar
        self.cte_frame = ttk.Frame(self.perceive_frame)
        self.cte_frame.pack(fill=tk.X)

        self.cte_gauge_frame = ttk.Frame(self.cte_frame)
        self.cte_gauge_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.cte_gauge_frame, text="Velocity CTE", font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.cte_gauge_frame, text="Position CTE", font=self.root.small_font).pack(side=tk.TOP)

        self.gauge_cte_vel = ValueGauge(self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_vel.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.gauge_cte_vel.set_marker(0)

        self.gauge_cte_steer = ValueGauge(self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_steer.set_marker(0)
        self.gauge_cte_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.gauge_cte_steer.set_marker(0)
        # ----

        self.vehicle_state_label = ttk.Label(self.perceive_frame,font=self.root.small_font, text="")
        self.vehicle_state_label.pack(side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)

        self.coordinates_label = ttk.Label(
            self.perceive_frame, text="Spawn Agent: Click on the plot."
        )
        self.coordinates_label.pack(side=tk.LEFT, padx=5, pady=5)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        ## Plan frame
        # ----------------------------------------------------------------------
        self.plan_frame = ttk.LabelFrame(self, text="Plan")
        self.plan_frame.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5, pady=5)
        self.global_button = ttk.Button(self.plan_frame, text="Global Replan")
        self.global_button.pack(side=tk.TOP, fill=tk.X, expand=True)
        

        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        self.global_tj_wp_entry = ttk.Entry(wp_frame, width=6)
        self.global_tj_wp_entry.insert(0, "0")
        self.global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(wp_frame, text=f"{len(self.root.exec.planner.global_trajectory.path_x)-1}").pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(self.plan_frame, text="Local Replan", command=self.replan).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # ----------------------------------------------------------------------
        ## Control Frame
        # ----------------------------------------------------------------------
        self.control_frame = ttk.LabelFrame(self, text="Control")
        self.control_frame.pack(fill=tk.X, expand=True, side=tk.LEFT)
        # Progress bar
        self.progress_frame = ttk.Frame(self.control_frame)
        self.progress_frame.pack(fill=tk.X)

        self.progress_label_frame = ttk.Frame(self.progress_frame)
        self.progress_label_frame.pack(side=tk.LEFT, padx=5)
        self.progress_label_acc = ttk.Label(
            self.progress_label_frame, text="Accel", font=self.root.small_font
        )
        self.progress_label_acc.pack(side=tk.TOP)
        self.progress_label_steer = ttk.Label(
            self.progress_label_frame, text="Steer", font=self.root.small_font
        )
        self.progress_label_steer.pack(side=tk.TOP)

        self.progressbar_acc = ValueGauge(
            self.progress_frame,
            min_value=self.root.exec.ego_state.min_acceleration,
            max_value=self.root.exec.ego_state.max_acceleration,
        )
        self.progressbar_acc.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.progressbar_acc.set_marker(0)

        self.progressbar_steer = ValueGauge(
            self.progress_frame,
            min_value=self.root.exec.ego_state.min_steering,
            max_value=self.root.exec.ego_state.max_steering,
        )
        self.progressbar_steer.set_marker(0)
        self.progressbar_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        # ----

        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_entry = ttk.Entry(dt_frame, width=5)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dt_frame, text="Control Step", command=self.step_control).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(dt_frame, text="Re-align", command=self.align_control).pack(
            side=tk.LEFT
        )  # Re-alignes with plan

        ttk.Button(self.control_frame, text="Steer L", command=self.step_steer_left).pack(
            side=tk.LEFT
        )
        ttk.Button(self.control_frame, text="Steer R", command=self.step_steer_right).pack(
            side=tk.LEFT
        )
        ttk.Button(self.control_frame, text="Accel.", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Decc.", command=self.step_dec).pack(side=tk.LEFT)

    # --------------------------------------------------------------------------------------------
    # -Plan---------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def set_waypoint(self):
        timestep_value = int(self.global_tj_wp_entry.get())
        self.root.exec.planner.reset(wp=timestep_value)
        self.root.update_ui()

    def replan(self):
        t1 = time.time()
        self.root.exec.planner.replan()
        t2 = time.time()
        log.info(f"Re-plan Time: {(t2-t1)*1000:.2f} ms")
        self.root.update_ui()

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.root.exec.planner.step_wp()
        log.info(f"Plan Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.global_tj_wp_entry.delete(0, tk.END)
        self.global_tj_wp_entry.insert(0, str(self.root.exec.planner.global_trajectory.next_wp - 1))
        self.root.update_ui()

    # --------------------------------------------------------------------------------------------
    # -Control------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def step_control(self):
        cmd = self.root.exec.controller.control(
            self.root.exec.ego_state, self.root.exec.planner.get_local_plan()
        )

        dt = float(self.dt_entry.get())
        self.root.exec.world.update_ego_state(state=self.root.exec.ego_state, cmd=cmd, dt=dt)
        self.root.update_ui()

    def align_control(self):
        self.root.exec.ego_state.x, self.root.exec.ego_state.y = self.root.exec.planner.location_xy
        self.root.update_ui()

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.root.exec.world.update_ego_state(
            state=self.root.exec.ego_state, cmd=ControlComand(steer=0.05), dt=dt
        )
        self.root.update_ui()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.root.exec.world.update_ego_state(
            state=self.root.exec.ego_state, cmd=ControlComand(steer=-0.05), dt=dt
        )
        self.root.update_ui()

    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.root.exec.world.update_ego_state(
            state=self.root.exec.ego_state, cmd=ControlComand(acc=8), dt=dt
        )
        self.root.update_ui()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.root.exec.world.update_ego_state(
            state=self.root.exec.ego_state, cmd=ControlComand(acc=-8), dt=dt
        )
        self.root.update_ui()
