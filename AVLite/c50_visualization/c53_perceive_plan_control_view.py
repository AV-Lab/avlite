from __future__ import annotations
import tkinter as tk
from tkinter import Variable, ttk
import time
import logging

from typing import TYPE_CHECKING
from c30_control.c31_base_controller import ControlComand
from c50_visualization.c58_ui_lib import ValueGauge
from c20_planning.c21_base_global_planner import BaseGlobalPlanner
from c20_planning.c24_base_local_planner import BaseLocalPlanner

if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

log = logging.getLogger(__name__)


class PerceivePlanControlView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root
        # ----------------------------------------------------------------------
        # Perceive Frame
        # ----------------------------------------------------------------------
        self.perceive_frame = ttk.LabelFrame(self, text="Perception")
        self.perceive_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # ----
        vehicle_state_label = ttk.Label(
            self.perceive_frame,
            font=self.root.small_font,
            textvariable=self.root.setting.vehicle_state,
            width=30,
            wraplength=235,  # Set wrap length in pixels
        )
        vehicle_state_label.pack(
            side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)

        self.coordinates_label = ttk.Label(
            self.perceive_frame, textvariable=self.root.setting.perception_status_text, width=30)
        self.coordinates_label.pack(side=tk.LEFT, padx=5, pady=5)
        # ----------------------------------------------------------------------


        # ----------------------------------------------------------------------
        # Plan frame
        # ----------------------------------------------------------------------
        self.plan_frame = ttk.LabelFrame(self, text="Planning")
        self.plan_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        # - Global -----
        global_frame = ttk.Frame(self.plan_frame)
        global_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            global_frame,
            text="Show Global Plan",
            command=self.root.toggle_plan_view,
            variable=self.root.setting.global_plan_view,
        ).pack(side=tk.LEFT)
        global_planner_dropdown_menu = ttk.Combobox(
            global_frame, textvariable=self.root.setting.global_planner_type, width=10)
        global_planner_dropdown_menu["values"] = tuple(BaseGlobalPlanner.registry.keys())
        global_planner_dropdown_menu.state(["readonly"])
        global_planner_dropdown_menu.pack(side=tk.LEFT)
        global_planner_dropdown_menu.bind("<<ComboboxSelected>>",
                                self.__on_dropdown_change)

        ttk.Button(global_frame, text="Global Replan").pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        # - Local -----
        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            wp_frame,
            text="Show Local Plan  ",
            command=self.root.toggle_plan_view,
            variable=self.root.setting.local_plan_view,
        ).pack(side=tk.LEFT)

        local_planner_dropdown_menu = ttk.Combobox(
            wp_frame, textvariable=self.root.setting.local_planner_type, width=10)
        local_planner_dropdown_menu["values"] = tuple(BaseLocalPlanner.registry.keys())
        local_planner_dropdown_menu.state(["readonly"])
        local_planner_dropdown_menu.pack(side=tk.LEFT)
        local_planner_dropdown_menu.bind("<<ComboboxSelected>>",
                                self.__on_dropdown_change)

        ttk.Button(wp_frame, text="Set Waypoint",
                   command=self.set_waypoint).pack(side=tk.LEFT)
        global_tj_wp_entry = ttk.Entry(
            wp_frame, width=6, textvariable=self.root.setting.current_wp)
        global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        global_tj_wp_entry.bind("<Return>", self.text_on_enter)
        ttk.Label(wp_frame, text=f"{len(self.root.exec.local_planner.global_trajectory.path_x)-1}").pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(self.plan_frame, text="Lap: ").pack(side=tk.LEFT, padx=5)
        ttk.Label(self.plan_frame, font=self.root.small_font,
                  textvariable=self.root.setting.lap).pack(side=tk.LEFT, padx=5)

        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(
            side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Local Replan",
                   command=self.replan).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ----------------------------------------------------------------------
        # Control Frame
        # ----------------------------------------------------------------------
        self.control_frame = ttk.LabelFrame(self, text="Control")
        self.control_frame.pack(fill=tk.X, expand=True, side=tk.LEFT)
        # Progress bars
        self.cte_frame = ttk.Frame(self.control_frame)
        self.cte_frame.pack(fill=tk.X)

        self.cte_gauge_frame = ttk.Frame(self.cte_frame)
        self.cte_gauge_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.cte_gauge_frame, text="Vel CTE",
                  font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.cte_gauge_frame, text="Pos CTE",
                  font=self.root.small_font).pack(side=tk.TOP)
        self.gauge_cte_vel = ValueGauge(
            self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_vel.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.gauge_cte_steer = ValueGauge(
            self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.progress_frame = ttk.Frame(self.control_frame)
        self.progress_frame.pack(fill=tk.X)

        self.progress_label_frame = ttk.Frame(self.progress_frame)
        self.progress_label_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.progress_label_frame, text="Accel",
                  font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.progress_label_frame, text="Steer",
                  font=self.root.small_font).pack(side=tk.TOP)

        self.gauge_acc = ValueGauge(
            self.progress_frame,
            min_value=self.root.exec.ego_state.min_acceleration,
            max_value=self.root.exec.ego_state.max_acceleration,
        )
        self.gauge_acc.pack(side=tk.TOP, fill=tk.X, expand=True)
        # self.progressbar_acc.set_marker(0)

        self.gauge_steer = ValueGauge(
            self.progress_frame,
            min_value=self.root.exec.ego_state.min_steering,
            max_value=self.root.exec.ego_state.max_steering,
        )
        # self.progressbar_steer.set_marker(0)
        self.gauge_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        # ----

        ttk.Button(self.control_frame, text="Step", command=self.step_control).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(self.control_frame, text="Align", width=4,
                   command=self.align_control).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="L", width=2,
                   command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="R", width=2,
                   command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accel.", width=4,
                   command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Dec.", width=4,
                   command=self.step_dec).pack(side=tk.LEFT)

        # Joystick
        if self.root.setting.enable_joystick:
            import pygame

            pygame.init()
            pygame.joystick.init()

            # Check for joystick
            if pygame.joystick.get_count() == 0:
                log.warning("No joystick connected")
                return

            # Initialize the first joystick
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

            self._controller_check_id = None
            self.start_controller_polling()

    # --------------------------------------------------------------------------------------------
    # -Plan---------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def set_waypoint(self):
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()
    
    def text_on_enter(self, event):
        widget = event.widget  # Get the widget that triggered the event
        text = widget.get()    # Retrieve the text from the widget
        self.root.validate_float_input(text)  # Validate the input
        log.debug("Text entered: %s", text)
        widget.tk_focusNext().focus_set()  # Move focus to the next widget
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()

    def replan(self):
        t1 = time.time()
        self.root.exec.local_planner.replan()
        t2 = time.time()
        log.info(f"Re-plan Time: {(t2-t1)*1000:.2f} ms")
        self.root.update_ui()

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.root.exec.local_planner.step_wp()
        log.info(f"Plan Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.root.setting.current_wp.set(str(self.root.exec.local_planner.global_trajectory.next_wp - 1))
        self.root.update_ui()

    def __on_dropdown_change(self, event):
        log.debug(
            f"Dropdown changed to: {self.root.setting.global_planner_type.get()}")
        self.root.reload_stack()
        self.root.global_plan_plot_view.update_plot_type()

    # --------------------------------------------------------------------------------------------
    # -Control------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def step_control(self):
        cmd = self.root.exec.controller.control(
            self.root.exec.ego_state, self.root.exec.local_planner.get_local_plan())

        self.root.exec.world.control_ego_state(
            cmd=cmd, dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def align_control(self):
        self.root.exec.ego_state.x, self.root.exec.ego_state.y = self.root.exec.local_planner.location_xy
        self.root.update_ui()

    def step_steer_left(self):
        self.root.exec.world.control_ego_state(cmd=ControlComand(
            steer=0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_steer_right(self):
        self.root.exec.world.control_ego_state(cmd=ControlComand(
            steer=-0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_acc(self):
        acc = 3
        self.root.exec.world.control_ego_state(
            cmd=ControlComand(acc=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_dec(self):
        acc = -3
        self.root.exec.world.control_ego_state(
            cmd=ControlComand(acc=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def start_controller_polling(self):
        """Start regular polling of the controller"""
        self.process_controller_input()
        # Schedule next check (every 50ms = 20fps)
        self._controller_check_id = self.after(
            50, self.start_controller_polling)

    def stop_controller_polling(self):
        """Stop controller polling"""
        if self._controller_check_id:
            self.after_cancel(self._controller_check_id)
            self._controller_check_id = None

    def process_controller_input(self):
        """Process Xbox controller input for steering and acceleration"""
        if not hasattr(self, "joystick") or self.joystick is None:
            return

        import pygame

        pygame.event.pump()  # Process pygame events

        left_stick_x = self.joystick.get_axis(0)
        right_trigger = self.joystick.get_axis(4)
        left_trigger = self.joystick.get_axis(5)

        # Apply deadzone to avoid drift
        if abs(left_stick_x) < 0.02:
            left_stick_x = 0

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"Left stick x: {left_stick_x}, Right trigger: {right_trigger}, Left trigger: {left_trigger}")

        # Scale inputs to control values
        # Negative for correct direction
        steering = -left_stick_x * self.root.exec.ego_state.max_steering
        acceleration = (right_trigger + 1) / 2 * \
            self.root.exec.ego_state.max_acceleration
        braking = (left_trigger + 1) / 2 * \
            self.root.exec.ego_state.min_acceleration

        # Apply controls if needed
        if abs(steering) > 0.01 or abs(acceleration) > 0.01 or abs(braking) > 0.01:
            cmd = ControlComand(steer=steering, acc=acceleration + braking)
            log.debug(f"Controller Command: {cmd}")
            self.root.exec.world.control_ego_state(
                cmd=cmd, dt=self.root.setting.sim_dt.get())
