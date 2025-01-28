from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import time

import logging
log = logging.getLogger(__name__)

class PerceivePlanControlView:
    def __init__(self, root: VisualizerApp):
        self.root = root
        # ----------------------------------------------------------------------
        # Percieve Plan Control Frame -----------------------------------------
        # ----------------------------------------------------------------------
        self.perceive_plan_control_frame = ttk.Frame(root)
        self.perceive_plan_control_frame.pack(fill=tk.X)
        # ----------------------------------------------------------------------
        ## Perceive Frame
        # ----------------------------------------------------------------------
        self.perceive_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Perceive")
        self.perceive_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.vehicle_state_label = ttk.Label(self.perceive_frame, text="")
        self.vehicle_state_label.pack(side=tk.TOP, expand=True, fill=tk.X, pady=5)

        self.coordinates_label = ttk.Label(self.perceive_frame, text="Spawn Agent: Click on the plot.")
        self.coordinates_label.pack(side=tk.LEFT, pady=5)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        ## Plan frame
        # ----------------------------------------------------------------------
        self.plan_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Plan (Manual)")
        self.plan_frame.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5, pady=5)

        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        self.global_tj_wp_entry = ttk.Entry(wp_frame, width=6)
        self.global_tj_wp_entry.insert(0, "0")
        self.global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(wp_frame, text=f"{len(self.root.exec.planner.global_trajectory.path_x)-1}").pack(side=tk.LEFT, padx=5)

        ttk.Button(self.plan_frame, text="Replan", command=self.replan).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ----------------------------------------------------------------------
        ## Control Frame
        # ----------------------------------------------------------------------
        self.control_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Control (Manual)")
        self.control_frame.pack(fill=tk.X, expand=True, side=tk.LEFT)
        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_entry = ttk.Entry(dt_frame, width=5)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dt_frame, text="Control Step", command=self.step_control).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dt_frame, text="Re-align", command=self.align_control).pack(side=tk.LEFT)  # Re-alignes with plan

        ttk.Button(self.control_frame, text="Steer Left", command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Steer Right", command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accelerate", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Deccelerate", command=self.step_dec).pack(side=tk.LEFT)

    # --------------------------------------------------------------------------------------------
    # -Plan---------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def set_waypoint(self):
        timestep_value = int(self.global_tj_wp_entry.get())
        self.root.exec.planner.reset(wp=timestep_value)
        self.root.plot_view.replot()

    def replan(self):
        t1 = time.time()
        self.root.exec.planner.replan()
        t2 = time.time()
        log.info(f"Re-plan Time: {(t2-t1)*1000:.2f} ms")
        self.root.plot_view.replot()

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.root.exec.planner.step_wp()
        log.info(f"Plan Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.global_tj_wp_entry.delete(0, tk.END)
        self.global_tj_wp_entry.insert(0, str(self.root.exec.planner.global_trajectory.next_wp - 1))
        self.root.plot_view.replot()

    # --------------------------------------------------------------------------------------------
    # -Control------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def step_control(self):
        d = self.root.exec.planner.traversed_d[-1]
        steer = self.root.exec.controller.control(d)

        dt = float(self.dt_entry.get())
        self.root.exec.update_ego_state(steering_angle=steer, dt=dt)
        self.root.plot_view.replot()

    def align_control(self):
        self.root.exec.ego_state.x = self.root.exec.planner.global_trajectory.path_x[self.root.exec.planner.global_trajectory.next_wp - 1]
        self.root.exec.ego_state.y = self.root.exec.planner.global_trajectory.path_y[self.root.exec.planner.global_trajectory.next_wp - 1]

        self.root.plot_view.replot()

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.root.exec.update_ego_state(dt=dt, steering_angle=0.1)
        self.root.plot_view.replot()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.root.exec.update_ego_state(dt=dt, steering_angle=-0.1)
        self.root.plot_view.replot()

    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.root.exec.update_ego_state(dt=dt, acceleration=8)
        self.root.plot_view.replot()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.root.exec.update_ego_state(dt=dt, acceleration=-8)
        self.root.plot_view.replot()
