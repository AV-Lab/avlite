from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import time
import logging
from datetime import datetime


from typing import TYPE_CHECKING

from avlite.c10_perception.c12_perception_strategy import (
    PerceptionStrategy, DetectionStrategy, TrackingStrategy, PredictionStrategy,
)
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c30_control.c32_control_strategy import ControlComand, ControlStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_visualization.c58_ui_lib import ValueGauge, ThemedInputDialog, attach_schema_tooltip
from avlite.c50_visualization.c59_settings import VisualizationSettings, DEFAULT_SUBSTRATEGY
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c60_common.c69_setting_utils import get_absolute_path, list_extensions
from avlite.c60_common.c61_capabilities import WorldCapability

if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

log = logging.getLogger(__name__)


class PerceivePlanControlView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root

        # Top bar: perceive / plan / control side by side
        top_bar = ttk.Frame(self)
        top_bar.pack(fill=tk.X)

        self.perceive_frame = PerceptionFrame(root=self.root, view=top_bar)
        self.perceive_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.plan_frame = PlanFrame(root=self.root, view=top_bar)
        self.plan_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.control_frame = ControlFrame(root=self.root, view=top_bar)
        self.control_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Extras row – hidden until the checkbox enables it
        self.perception_extras_frame = PerceptionExtrasFrame(root=self.root, view=self)
        self.root.setting.show_perception_extras.trace_add("write", lambda *_: self.toggle_perception_extras())
        self.toggle_perception_extras()  # sync initial checkbox state

    def toggle_perception_extras(self):
        if self.root.setting.show_perception_extras.get():
            self.perception_extras_frame.pack(fill=tk.X, expand=True)
        else:
            self.perception_extras_frame.pack_forget()

    def reset(self):
        """Update data in the view."""
        self.perceive_frame.update_data()
        self.plan_frame.update_data()
        self.control_frame.update_data()


# --------------------------------------------------------------------------------------------
# -Perception---------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
class PerceptionFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view: ttk.Frame):
        super().__init__(view, text="Perception")
        self.root = root

        # Row 0: main perception dropdown + Show checkbox + Extras checkbox
        self.perception_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.perception_type, state="readonly", width=14)
        self.perception_dropdown_menu["values"] = list(PerceptionStrategy.registry.keys())
        self.perception_dropdown_menu.bind("<<ComboboxSelected>>", self._on_perception_selected)
        self.perception_dropdown_menu.grid(row=0, column=0, sticky="ew", padx=2)
        attach_schema_tooltip(self.perception_dropdown_menu, ExecutionSettings, "c40_perception")

        show_occ = ttk.Checkbutton(self, text="Show", variable=self.root.setting.show_occupancy_flow)
        show_occ.grid(row=0, column=1, padx=2)
        attach_schema_tooltip(show_occ, VisualizationSettings, "show_occupancy_flow")
        extras_cb = ttk.Checkbutton(
            self, text="Extras", variable=self.root.setting.show_perception_extras,
            command=lambda: self.root.perceive_plan_control_view.toggle_perception_extras(),
        )
        extras_cb.grid(row=0, column=2, padx=2)
        attach_schema_tooltip(extras_cb, VisualizationSettings, "show_perception_extras")

        # Rows 1-3: pipeline sub-strategy widgets (shown only for PerceptionPipeline)
        self._lbl_detect = ttk.Label(self, text="Detect:")
        self._lbl_detect.grid(row=1, column=0, sticky="e", padx=(5, 0))
        self.detection_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.detection_strategy_type, state="readonly")
        self.detection_dropdown_menu["values"] = (
            ((DEFAULT_SUBSTRATEGY,) if WorldCapability.GT_DETECTION in self.root.exec.world.capabilities else ())
            + tuple(DetectionStrategy.registry.keys())
        )
        self.detection_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.detection_dropdown_menu.grid(row=1, column=1, columnspan=2, sticky="ew")

        self._lbl_track = ttk.Label(self, text="Track:")
        self._lbl_track.grid(row=2, column=0, sticky="e", padx=(5, 0))
        self.tracking_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.tracking_strategy_type, state="readonly")
        self.tracking_dropdown_menu["values"] = (
            ((DEFAULT_SUBSTRATEGY,) if WorldCapability.GT_TRACKING in self.root.exec.world.capabilities else ())
            + tuple(TrackingStrategy.registry.keys())
        )
        self.tracking_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.tracking_dropdown_menu.grid(row=2, column=1, columnspan=2, sticky="ew")

        self._lbl_predict = ttk.Label(self, text="Predict:")
        self._lbl_predict.grid(row=3, column=0, sticky="e", padx=(5, 0))
        self.prediction_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.prediction_strategy_type, state="readonly")
        self.prediction_dropdown_menu["values"] = (DEFAULT_SUBSTRATEGY,) + tuple(PredictionStrategy.registry.keys())
        self.prediction_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.prediction_dropdown_menu.grid(row=3, column=1, columnspan=2, sticky="ew")
        attach_schema_tooltip(self.detection_dropdown_menu, PerceptionSettings, "c12_detection_strategy")
        attach_schema_tooltip(self.tracking_dropdown_menu, PerceptionSettings, "c12_tracking_strategy")
        attach_schema_tooltip(self.prediction_dropdown_menu, PerceptionSettings, "c12_prediction_strategy")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self._pipeline_widgets = [
            self._lbl_detect, self.detection_dropdown_menu,
            self._lbl_track, self.tracking_dropdown_menu,
            self._lbl_predict, self.prediction_dropdown_menu,
        ]

        self.root.setting.perception_type.trace_add("write", lambda *_: self._update_pipeline_visibility())
        self._update_pipeline_visibility()

    def _on_perception_selected(self, event=None):
        self._update_pipeline_visibility()
        self.root.reload_stack(reload_code=False)

    def _update_pipeline_visibility(self):
        from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
        is_pipeline = self.root.setting.perception_type.get() == PerceptionPipeline.__name__
        for w in self._pipeline_widgets:
            if is_pipeline:
                w.grid()
            else:
                w.grid_remove()

    def update_data(self):
        """Update data in the perception frame."""
        core_strategies = set(PerceptionStrategy.registry.keys())
        allowed_default_extensions = set(PerceptionStrategy.registry.keys()) & set(ExecutionSettings.c40_default_extensions)
        community_prefixes = tuple(f"avlite.extensions.{a}" for a in ExecutionSettings.c40_community_plugins.keys())
        allowed_community_extensions = {
            n for n, c in PerceptionStrategy.registry.items()
            if community_prefixes and c.__module__.startswith(community_prefixes)
        }
        data = sorted(core_strategies | allowed_default_extensions | allowed_community_extensions)
        log.warning(f"allowed_default_extensions: {allowed_default_extensions}, allowed_community_extensions: {allowed_community_extensions}")
        log.warning(f"final Strategies: {data}")

        self.perception_dropdown_menu["values"] = tuple(data)
        _caps = self.root.exec.world.capabilities
        self.detection_dropdown_menu["values"] = (
            ((DEFAULT_SUBSTRATEGY,) if WorldCapability.GT_DETECTION in _caps else ())
            + tuple(DetectionStrategy.registry.keys())
        )
        self.tracking_dropdown_menu["values"] = (
            ((DEFAULT_SUBSTRATEGY,) if WorldCapability.GT_TRACKING in _caps else ())
            + tuple(TrackingStrategy.registry.keys())
        )
        self.prediction_dropdown_menu["values"] = (DEFAULT_SUBSTRATEGY,) + tuple(PredictionStrategy.registry.keys())
        self._update_pipeline_visibility()


class PerceptionExtrasFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view: ttk.Frame):
        super().__init__(view, text="Perception Extras")
        self.root = root

        ttk.Label(self, text="Localization:").pack(side=tk.LEFT, padx=(5, 2))
        self.localization_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.localization_type, state="readonly", width=12)
        self.localization_dropdown_menu["values"] = (
            ((DEFAULT_SUBSTRATEGY,) if WorldCapability.GT_LOCALIZATION in self.root.exec.world.capabilities else ())
            + tuple(LocalizationStrategy.registry.keys())
        )
        self.localization_dropdown_menu.set(self.root.setting.localization_type.get() or DEFAULT_SUBSTRATEGY)
        self.localization_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.localization_dropdown_menu.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        attach_schema_tooltip(self.localization_dropdown_menu, ExecutionSettings, "c40_localization")

        ttk.Label(self, text="Mapping:").pack(side=tk.LEFT, padx=(5, 2))
        self.mapping_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.mapping_type, state="readonly", width=12)
        self.mapping_dropdown_menu["values"] = (DEFAULT_SUBSTRATEGY,) + tuple(MappingStrategy.registry.keys())
        self.mapping_dropdown_menu.set(self.root.setting.mapping_type.get() or DEFAULT_SUBSTRATEGY)
        self.mapping_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.mapping_dropdown_menu.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        attach_schema_tooltip(self.mapping_dropdown_menu, ExecutionSettings, "c40_mapping")

# --------------------------------------------------------------------------------------------
# -Plan---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
class PlanFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view:ttk.Frame):
        super().__init__(view, text="Planning")
        self.root = root
        
        # self.plan_frame = ttk.LabelFrame(self, text="Planning")
        # self.plan_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # - Global -----
        global_frame = ttk.Frame(self)
        global_frame.pack(fill=tk.X)
        ttk.Label(global_frame, text="Global: ").pack(side=tk.LEFT, padx=5)
        global_show = ttk.Checkbutton(global_frame, text="Show", command=self.root.update_views, variable=self.root.setting.global_plan_view)
        global_show.pack(side=tk.LEFT)
        attach_schema_tooltip(global_show, VisualizationSettings, "global_plan_view")
        self.global_planner_dropdown_menu = ttk.Combobox(global_frame, textvariable=self.root.setting.global_planner_type, width=10)
        self.global_planner_dropdown_menu["values"] = tuple(GlobalPlannerStrategy.registry.keys())
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        attach_schema_tooltip(self.global_planner_dropdown_menu, ExecutionSettings, "c40_global_planner")

        ttk.Button(global_frame, text="Global Replan", command=self.replan_global).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(global_frame, text="⬇", command=self.save_global_plan, width=3).pack(side=tk.LEFT)

        # - Local -----
        wp_frame = ttk.Frame(self)
        wp_frame.pack(fill=tk.X)
        # ttk.Separator(wp_frame, orient='horizontal').pack(side=tk.TOP,fill='x', pady=2)
        ttk.Label(wp_frame, text="Local:   ").pack(side=tk.LEFT, padx=5)
        local_show = ttk.Checkbutton(wp_frame, text="Show  ", command=self.root.update_views, variable=self.root.setting.local_plan_view)
        local_show.pack(side=tk.LEFT)
        attach_schema_tooltip(local_show, VisualizationSettings, "local_plan_view")

        self.local_planner_dropdown_menu = ttk.Combobox(wp_frame, textvariable=self.root.setting.local_planner_type, width=10)
        self.local_planner_dropdown_menu["values"] = tuple(LocalPlanningStrategy.registry.keys())
        self.local_planner_dropdown_menu.state(["readonly"])
        self.local_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.local_planner_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        attach_schema_tooltip(self.local_planner_dropdown_menu, ExecutionSettings, "c40_local_planner")

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        global_tj_wp_entry = ttk.Entry( wp_frame, width=6, textvariable=self.root.setting.current_wp)
        global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        global_tj_wp_entry.bind("<Return>", self.text_on_enter)
        ttk.Label(wp_frame, text=f"{len(self.root.exec.local_planner.global_trajectory.path_x)-1}").pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(self, text="Lap: ").pack(side=tk.LEFT, padx=5)
        ttk.Label(self, font=self.root.small_font,
                  textvariable=self.root.setting.lap).pack(side=tk.LEFT, padx=5)

        ttk.Button(self, text="◀️", command=self.step_waypoint_back, width=2).pack(side=tk.LEFT)
        ttk.Button(self, text="▶", command=self.step_plan, width=2).pack(side=tk.LEFT)
        ttk.Button(self, text="Align", command=self.align_plan, width=4).pack(side=tk.LEFT)
        ttk.Button(self, text="Local Replan", command=self.replan).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update_data(self):
        """Update data in the plan frame."""
        self.local_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.local_planner_dropdown_menu["values"] = tuple(LocalPlanningStrategy.registry.keys())
        self.global_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.global_planner_dropdown_menu["values"] = tuple(GlobalPlannerStrategy.registry.keys())

    def set_waypoint(self):
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()
    def step_waypoint_back(self):
        """ Step back to the previous waypoint in the local planner."""
        self.root.setting.current_wp.set(str(int(self.root.setting.current_wp.get()) - 1))
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

    def replan_global(self):
        self.root.exec.replan_global()
        self.root.global_plan_plot_view.global_plot.reset()
        self.root.local_plan_plot_view.reset()
        self.root.global_plan_plot_view.plot()
        self.root.local_plan_plot_view.plot()
        self.root.update_ui()

    def save_global_plan(self):
        self.root.exec_visualize_view.stop_exec()
        default_name = f"data/{datetime.now().strftime('%Y%m%d_%H%M%S')}_global_plan.json"
        dialog = ThemedInputDialog(self.root, "Save Global Plan", "File name:", initial=default_name)
        fname = dialog.result
        if not fname:
            return
        try:
            self.root.exec.local_planner.global_plan.to_file(get_absolute_path(fname, for_write=True))
        except OSError as e:
            messagebox.showerror("Save Failed", str(e), parent=self)

    def align_plan(self):
        log.debug("Aligning plan with current ego state")
        self.root.exec.local_planner.step(self.root.exec.world.get_ego_state())
        self.root.update_ui()

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.root.exec.local_planner.step_wp()
        log.info(f"Plan Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.root.setting.current_wp.set(str(self.root.exec.local_planner.global_trajectory.next_wp - 1))
        self.root.update_ui()



# --------------------------------------------------------------------------------------------
# -Control------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
class ControlFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Control")
        self.root = root

        # buttons
        control_button_frame = ttk.Frame(self)
        control_button_frame.pack(fill=tk.X, expand=True)
        self.controller_dropdown_menu = ttk.Combobox(control_button_frame, textvariable=self.root.setting.controller_type, width=10)
        self.controller_dropdown_menu["values"] = tuple(ControlStrategy.registry.keys())
        self.controller_dropdown_menu.state(["readonly"])
        self.controller_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.controller_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        attach_schema_tooltip(self.controller_dropdown_menu, ExecutionSettings, "c40_controller")

        ttk.Button(control_button_frame, text="Step", command=self.step_control).pack( side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(control_button_frame, text="Align", width=4, command=self.align_control).pack(side=tk.LEFT)
        ttk.Button(control_button_frame, text="◀️ ", width=2, command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(control_button_frame, text="▶", width=2, command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(control_button_frame, text="▲", width=2, command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(control_button_frame, text="▼", width=2, command=self.step_dec).pack(side=tk.LEFT)

        #################
        # Progress bars
        #################
        self.cte_frame = ttk.Frame(self)
        self.cte_frame.pack(fill=tk.X)

        self.cte_gauge_frame = ttk.Frame(self.cte_frame)
        self.cte_gauge_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.cte_gauge_frame, text="Vel CTE", font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.cte_gauge_frame, text="Pos CTE", font=self.root.small_font).pack(side=tk.TOP)
        self.gauge_cte_vel = ValueGauge( self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_vel.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.gauge_cte_steer = ValueGauge( self.cte_frame, min_value=-20, max_value=20)
        self.gauge_cte_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X)

        self.progress_label_frame = ttk.Frame(self.progress_frame)
        self.progress_label_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.progress_label_frame, text="Accel", font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.progress_label_frame, text="Steer", font=self.root.small_font).pack(side=tk.TOP)

        self.gauge_acc = ValueGauge( self.progress_frame,
            min_value=self.root.exec.controller.ego_min_acceleration,
            max_value=self.root.exec.controller.ego_max_acceleration,
        )
        self.gauge_acc.pack(side=tk.TOP, fill=tk.X, expand=True)
        # self.progressbar_acc.set_marker(0)

        self.gauge_steer = ValueGauge( self.progress_frame,
            min_value=self.root.exec.controller.ego_min_steering,
            max_value=self.root.exec.controller.ego_max_steering,
        )
        # self.progressbar_steer.set_marker(0)
        self.gauge_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        # ----


        self.setup_joystick()
    
    def update_data(self):
        """Update data in the control frame."""
        self.controller_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.controller_dropdown_menu["values"] = tuple(ControlStrategy.registry.keys())

    def setup_joystick(self):
        try:
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

                self.__controller_check_id = None
                self.start_controller_polling()
        except Exception as e:
            log.error(f"Error initializing joystick: {e}")


    def step_control(self):
        cmd = self.root.exec.controller.control(
            self.root.exec.ego_state, self.root.exec.local_planner.get_local_plan())

        self.root.exec.world.control_ego_state(
            cmd=cmd, dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def align_control(self):
        self.root.exec.ego_state.x, self.root.exec.ego_state.y = self.root.exec.local_planner.location_xy
        self.root.exec.controller.reset()
        self.root.update_ui()

    def step_steer_left(self):
        log.debug("Steer right")
        self.root.exec.world.control_ego_state(cmd=ControlComand(
            steer=0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_steer_right(self):
        log.debug("Steer right")
        self.root.exec.world.control_ego_state(cmd=ControlComand(
            steer=-0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def reset_steer(self):
        log.debug("Reset steer")
        self.root.exec.world.control_ego_state(cmd=ControlComand(
            steer=0), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_acc(self):
        acc = 3
        self.root.exec.world.control_ego_state(
            cmd=ControlComand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_dec(self):
        acc = -3
        self.root.exec.world.control_ego_state(
            cmd=ControlComand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def start_controller_polling(self):
        """Start regular polling of the controller"""
        self.process_controller_input()
        # Schedule next check (every 50ms = 20fps)
        self.__controller_check_id = self.after(50, self.start_controller_polling)

    def stop_controller_polling(self):
        """Stop controller polling"""
        if self.__controller_check_id:
            self.after_cancel(self.__controller_check_id)
            self.__controller_check_id = None

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
        steering = -left_stick_x * self.root.exec.controller.ego_max_steering
        acceleration = (right_trigger + 1) / 2 * \
            self.root.exec.controller.ego_max_acceleration
        braking = (left_trigger + 1) / 2 * \
            self.root.exec.controller.ego_min_acceleration

        # Apply controls if needed
        if abs(steering) > 0.01 or abs(acceleration) > 0.01 or abs(braking) > 0.01:
            cmd = ControlComand(steer=steering, acceleration=acceleration + braking)
            log.debug(f"Controller Command: {cmd}")
            self.root.exec.world.control_ego_state(
                cmd=cmd, dt=self.root.setting.sim_dt.get())
