from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from avlite.c10_perception.c12_perception_strategy import (
    PerceptionStrategy, DetectionStrategy, TrackingStrategy, PredictionStrategy,
    PerceptionPipeline,
)
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import (
    LocalBehavioralPlanningStrategy,
    LocalPathPlanningStrategy,
    LocalPlanningPipeline,
    LocalPlanningStrategy,
    LocalVelocityPlanningStrategy,
)
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c49_settings import (
    ExecutionSettings,
    is_capability_provided,
)
from avlite.c60_apps.c69_settings import AppSettings
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import (
    ValueGauge,
    DataPicker,
    attach_schema_tooltip,
    attach_tooltip,
    BUTTON_TOOLTIPS,
    ThemedListPickerDialog,
    update_schema_tooltip,
)
from avlite.plugins.p60_visualizer_tk.settings import VisualizationSettings
from avlite.c60_apps.c63_plugins import plugin_module_prefix
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability
from avlite.c60_apps.c68_paths import DataPaths

if TYPE_CHECKING:
    from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp

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
        self.root.setting.p67_show_perception_extras.trace_add("write", lambda *_: self.toggle_perception_extras())
        self.toggle_perception_extras()  # sync initial checkbox state

    def toggle_perception_extras(self):
        if self.root.setting.p67_show_perception_extras.get():
            self.perception_extras_frame.pack(fill=tk.X, expand=True)
        else:
            self.perception_extras_frame.pack_forget()

    def reset(self):
        """Update data in the view."""
        self.perceive_frame.update_data()
        if self.root.exec is not None:
            stack_caps = self.root.exec.world.stack_capabilities
            self.perception_extras_frame.localization_dropdown_menu["values"] = (
                (("",) if StackCapability.LOCALIZATION in stack_caps else ())
                + tuple(LocalizationStrategy.registry.keys())
            )
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

        show_occ = ttk.Checkbutton(self, text="Show", variable=self.root.setting.p67_show_occupancy_flow)
        show_occ.grid(row=0, column=1, padx=2)
        attach_schema_tooltip(show_occ, VisualizationSettings, "p67_show_occupancy_flow")
        extras_cb = ttk.Checkbutton(
            self, text="Extras", variable=self.root.setting.p67_show_perception_extras,
            command=lambda: self.root.perceive_plan_control_view.toggle_perception_extras(),
        )
        extras_cb.grid(row=0, column=2, padx=2)
        attach_schema_tooltip(extras_cb, VisualizationSettings, "p67_show_perception_extras")

        # Rows 1-3: pipeline sub-strategy widgets (shown only for PerceptionPipeline)
        self._lbl_detect = ttk.Label(self, text="Detect:")
        self._lbl_detect.grid(row=1, column=0, sticky="e", padx=(5, 0))
        self.detection_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.detection_strategy_type, state="readonly")
        self.detection_dropdown_menu["values"] = tuple(DetectionStrategy.registry.keys())
        self.detection_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        self.detection_dropdown_menu.grid(row=1, column=1, columnspan=2, sticky="ew")

        self._lbl_track = ttk.Label(self, text="Track:")
        self._lbl_track.grid(row=2, column=0, sticky="e", padx=(5, 0))
        self.tracking_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.tracking_strategy_type, state="readonly")
        self.tracking_dropdown_menu["values"] = tuple(TrackingStrategy.registry.keys())
        self.tracking_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        self.tracking_dropdown_menu.grid(row=2, column=1, columnspan=2, sticky="ew")

        self._lbl_predict = ttk.Label(self, text="Predict:")
        self._lbl_predict.grid(row=3, column=0, sticky="e", padx=(5, 0))
        self.prediction_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.prediction_strategy_type, state="readonly")
        self.prediction_dropdown_menu["values"] = ("",) + tuple(PredictionStrategy.registry.keys())
        self.prediction_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
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
        is_pipeline = self.root.setting.perception_type.get() == PerceptionPipeline.__name__
        for w in self._pipeline_widgets:
            if is_pipeline:
                w.grid()
            else:
                w.grid_remove()

    def update_data(self):
        """Update data in the perception frame."""
        core_strategies = set(PerceptionStrategy.registry.keys())
        allowed_default_plugins = set(PerceptionStrategy.registry.keys()) & set(AppSettings.c62_default_plugins)
        community_prefixes = tuple(plugin_module_prefix(a) for a in AppSettings.c62_community_plugins.keys())
        allowed_community_plugins = {
            n for n, c in PerceptionStrategy.registry.items()
            if community_prefixes and c.__module__.startswith(community_prefixes)
        }
        data = sorted(core_strategies | allowed_default_plugins | allowed_community_plugins)
        log.warning("allowed_default_plugins: %s, allowed_community_plugins: %s", allowed_default_plugins, allowed_community_plugins)
        log.warning(f"final Strategies: {data}")

        self.perception_dropdown_menu["values"] = tuple(data)
        if self.root.exec is None:
            return
        _stack_caps = self.root.exec.world.stack_capabilities
        self.detection_dropdown_menu["values"] = (
            (("",) if StackCapability.DETECTION in _stack_caps else ())
            + tuple(DetectionStrategy.registry.keys())
        )
        self.tracking_dropdown_menu["values"] = (
            (("",) if StackCapability.TRACKING in _stack_caps else ())
            + tuple(TrackingStrategy.registry.keys())
        )
        self.prediction_dropdown_menu["values"] = ("",) + tuple(PredictionStrategy.registry.keys())
        self._update_pipeline_visibility()


class PerceptionExtrasFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view: ttk.Frame):
        super().__init__(view, text="Perception Extras")
        self.root = root

        ttk.Label(self, text="Localization:").pack(side=tk.LEFT, padx=(5, 2))
        self.localization_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.localization_type, state="readonly", width=12)
        self.localization_dropdown_menu["values"] = tuple(LocalizationStrategy.registry.keys())
        self.localization_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.localization_dropdown_menu.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        attach_schema_tooltip(self.localization_dropdown_menu, ExecutionSettings, "c40_localization")

        ttk.Label(self, text="Mapping:").pack(side=tk.LEFT, padx=(5, 2))
        self.mapping_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.mapping_type, state="readonly", width=12)
        self.mapping_dropdown_menu["values"] = ("",) + tuple(MappingStrategy.registry.keys())
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
        global_show = ttk.Checkbutton(global_frame, text="Global", command=self.root.update_views, variable=self.root.setting.p67_global_plan_view)
        global_show.pack(side=tk.LEFT)
        attach_schema_tooltip(global_show, VisualizationSettings, "p67_global_plan_view")
        self.global_planner_dropdown_menu = ttk.Combobox(global_frame, textvariable=self.root.setting.global_planner_type, width=10)
        self.global_planner_dropdown_menu["values"] = tuple(GlobalPlannerStrategy.registry.keys())
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        attach_schema_tooltip(self.global_planner_dropdown_menu, ExecutionSettings, "c40_global_planner")

        btn_global_replan = ttk.Button(global_frame, text="Global Replan", command=self.replan_global)
        btn_global_replan.pack(side=tk.LEFT, fill=tk.X, expand=True)
        attach_tooltip(btn_global_replan, BUTTON_TOOLTIPS["plan_global_replan"])
        btn_save_global = ttk.Button(global_frame, text="⬇", command=self.save_global_plan, width=3)
        btn_save_global.pack(side=tk.LEFT)
        attach_tooltip(btn_save_global, BUTTON_TOOLTIPS["plan_save_global"])

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=2)

        # - Local -----
        wp_frame = ttk.Frame(self)
        wp_frame.pack(fill=tk.X)
        local_show = ttk.Checkbutton(wp_frame, text="Local", command=self.root.update_views, variable=self.root.setting.p67_local_plan_view)
        local_show.pack(side=tk.LEFT)
        attach_schema_tooltip(local_show, VisualizationSettings, "p67_local_plan_view")

        self.local_planner_dropdown_menu = ttk.Combobox(wp_frame, textvariable=self.root.setting.local_planner_type, width=10)
        self.local_planner_dropdown_menu["values"] = tuple(LocalPlanningStrategy.registry.keys())
        self.local_planner_dropdown_menu.state(["readonly"])
        self.local_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.local_planner_dropdown_menu.bind("<<ComboboxSelected>>", self._on_local_planner_selected)
        attach_schema_tooltip(self.local_planner_dropdown_menu, ExecutionSettings, "c40_local_planner")

        btn_set_wp = ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint)
        btn_set_wp.pack(side=tk.LEFT)
        attach_tooltip(btn_set_wp, BUTTON_TOOLTIPS["plan_set_waypoint"])
        global_tj_wp_entry = ttk.Entry( wp_frame, width=6, textvariable=self.root.setting.current_wp)
        global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        global_tj_wp_entry.bind("<Return>", self.text_on_enter)
        self._wp_count_label = ttk.Label(wp_frame, text="0")
        self._wp_count_label.pack(side=tk.LEFT, padx=5)

        self._lap_label = ttk.Label(self, text="Lap: ")
        self._local_sub_frame = ttk.Frame(self)
        self._local_sub_frame.pack(fill=tk.X)
        local_g = ttk.Checkbutton(self._local_sub_frame, text="G", variable=self.root.setting.p67_show_local_global_view, command=self.root.update_ui)
        local_g.pack(side=tk.LEFT)
        local_f = ttk.Checkbutton(self._local_sub_frame, text="F", variable=self.root.setting.p67_show_local_frenet_view, command=self.root.update_ui)
        local_f.pack(side=tk.LEFT)
        attach_schema_tooltip(local_g, VisualizationSettings, "p67_show_local_global_view")
        attach_schema_tooltip(local_f, VisualizationSettings, "p67_show_local_frenet_view")

        self._lbl_behavior = ttk.Label(self._local_sub_frame, text="Behavior:")
        self.behavioral_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.behavioral_strategy_type, state="readonly", width=8)
        self.behavioral_dropdown_menu["values"] = ("",) + tuple(LocalBehavioralPlanningStrategy.registry.keys())
        self.behavioral_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        attach_schema_tooltip(self.behavioral_dropdown_menu, PlanningSettings, "c23_behavioral_strategy")
        self._lbl_path = ttk.Label(self._local_sub_frame, text="Path:")
        self.path_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.path_strategy_type, state="readonly", width=8)
        self.path_dropdown_menu["values"] = ("",) + tuple(LocalPathPlanningStrategy.registry.keys())
        self.path_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        attach_schema_tooltip(self.path_dropdown_menu, PlanningSettings, "c23_path_strategy")
        self._lbl_speed = ttk.Label(self._local_sub_frame, text="Speed:")
        self.velocity_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.velocity_strategy_type, state="readonly", width=8)
        self.velocity_dropdown_menu["values"] = ("",) + tuple(LocalVelocityPlanningStrategy.registry.keys())
        self.velocity_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        attach_schema_tooltip(self.velocity_dropdown_menu, PlanningSettings, "c23_velocity_strategy")
        self._pipeline_widgets = (
            (self._lbl_behavior, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.behavioral_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (self._lbl_path, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.path_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (self._lbl_speed, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.velocity_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
        )

        self.root.setting.local_planner_type.trace_add("write", lambda *_: self._update_pipeline_visibility())
        self._update_pipeline_visibility()

        self._lap_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(self, font=self.root.small_font,
                  textvariable=self.root.setting.lap).pack(side=tk.LEFT, padx=5)

        btn_wp_back = ttk.Button(self, text="◀️", command=self.step_waypoint_back, width=2)
        btn_wp_back.pack(side=tk.LEFT)
        attach_tooltip(btn_wp_back, BUTTON_TOOLTIPS["plan_wp_back"])
        btn_plan_step = ttk.Button(self, text="▶", command=self.step_plan, width=2)
        btn_plan_step.pack(side=tk.LEFT)
        attach_tooltip(btn_plan_step, BUTTON_TOOLTIPS["plan_step"])
        btn_plan_align = ttk.Button(self, text="Align", command=self.align_plan, width=4)
        btn_plan_align.pack(side=tk.LEFT)
        attach_tooltip(btn_plan_align, BUTTON_TOOLTIPS["plan_align"])
        btn_local_replan = ttk.Button(self, text="Local Replan", command=self.replan)
        btn_local_replan.pack(side=tk.LEFT, fill=tk.X, expand=True)
        attach_tooltip(btn_local_replan, BUTTON_TOOLTIPS["plan_local_replan"])

    def _on_local_planner_selected(self, event=None):
        self._update_pipeline_visibility()
        self.root.reload_stack(reload_code=False)

    def _update_pipeline_visibility(self):
        show = self.root.setting.local_planner_type.get() == LocalPlanningPipeline.__name__
        for widget, pack_kwargs in self._pipeline_widgets:
            if show:
                widget.pack(**pack_kwargs)
            else:
                widget.pack_forget()

    def update_data(self):
        """Update data in the plan frame."""
        self.local_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.local_planner_dropdown_menu["values"] = tuple(LocalPlanningStrategy.registry.keys())
        self.global_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.global_planner_dropdown_menu["values"] = tuple(GlobalPlannerStrategy.registry.keys())
        self.behavioral_dropdown_menu["values"] = ("",) + tuple(LocalBehavioralPlanningStrategy.registry.keys())
        self.path_dropdown_menu["values"] = ("",) + tuple(LocalPathPlanningStrategy.registry.keys())
        self.velocity_dropdown_menu["values"] = ("",) + tuple(LocalVelocityPlanningStrategy.registry.keys())
        self._update_pipeline_visibility()
        if self.root.exec is not None:
            self._wp_count_label.config(
                text=f"{len(self.root.exec.local_planner.global_trajectory.path_x) - 1}"
            )

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
        self.root.local_plan_plot_view.reset()
        self.root.global_plan_plot_view.plot()
        self.root.local_plan_plot_view.plot()
        self.root.update_ui()

    def save_global_plan(self):
        self.root.exec_visualize_view.stop_exec()
        data_dir = DataPaths.user_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_global_plan.json"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Global Plan",
            initialdir=str(data_dir),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.root.exec.global_planner.global_plan.to_file(path)
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

        btn_control_step = ttk.Button(control_button_frame, text="Step", command=self.step_control)
        btn_control_step.pack(side=tk.LEFT, fill=tk.X, expand=True)
        attach_tooltip(btn_control_step, BUTTON_TOOLTIPS["control_step"])
        btn_control_align = ttk.Button(control_button_frame, text="Align", width=4, command=self.align_control)
        btn_control_align.pack(side=tk.LEFT)
        attach_tooltip(btn_control_align, BUTTON_TOOLTIPS["control_align"])
        btn_steer_left = ttk.Button(control_button_frame, text="◀️ ", width=2, command=self.step_steer_left)
        btn_steer_left.pack(side=tk.LEFT)
        attach_tooltip(btn_steer_left, BUTTON_TOOLTIPS["control_steer_left"])
        btn_steer_right = ttk.Button(control_button_frame, text="▶", width=2, command=self.step_steer_right)
        btn_steer_right.pack(side=tk.LEFT)
        attach_tooltip(btn_steer_right, BUTTON_TOOLTIPS["control_steer_right"])
        btn_accel = ttk.Button(control_button_frame, text="▲", width=2, command=self.step_acc)
        btn_accel.pack(side=tk.LEFT)
        attach_tooltip(btn_accel, BUTTON_TOOLTIPS["control_accel"])
        btn_decel = ttk.Button(control_button_frame, text="▼", width=2, command=self.step_dec)
        btn_decel.pack(side=tk.LEFT)
        attach_tooltip(btn_decel, BUTTON_TOOLTIPS["control_decel"])

        #################
        # Progress bars
        #################
        self.cte_frame = ttk.Frame(self)
        self.cte_frame.pack(fill=tk.X)

        self.cte_gauge_frame = ttk.Frame(self.cte_frame)
        self.cte_gauge_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.cte_gauge_frame, text="Vel CTE", font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.cte_gauge_frame, text="Pos CTE", font=self.root.small_font).pack(side=tk.TOP)
        self.gauge_cte_vel = ValueGauge(self.cte_frame, min_value=-20, max_value=20, dpi_scale=self.root._dpi_scale)
        self.gauge_cte_vel.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.gauge_cte_steer = ValueGauge(self.cte_frame, min_value=-20, max_value=20, dpi_scale=self.root._dpi_scale)
        self.gauge_cte_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X)

        self.progress_label_frame = ttk.Frame(self.progress_frame)
        self.progress_label_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.progress_label_frame, text="Accel", font=self.root.small_font).pack(side=tk.TOP)
        ttk.Label(self.progress_label_frame, text="Steer", font=self.root.small_font).pack(side=tk.TOP)

        self.gauge_acc = ValueGauge(
            self.progress_frame,
            min_value=ControlSettings.c32_ego_min_acceleration,
            max_value=ControlSettings.c32_ego_max_acceleration,
            dpi_scale=self.root._dpi_scale,
        )
        self.gauge_acc.pack(side=tk.TOP, fill=tk.X, expand=True)
        # self.progressbar_acc.set_marker(0)

        self.gauge_steer = ValueGauge(
            self.progress_frame,
            min_value=ControlSettings.c32_ego_min_steering,
            max_value=ControlSettings.c32_ego_max_steering,
            dpi_scale=self.root._dpi_scale,
        )
        # self.progressbar_steer.set_marker(0)
        self.gauge_steer.pack(side=tk.TOP, fill=tk.X, expand=True)
        # ----

    def update_data(self):
        """Update data in the control frame."""
        self.controller_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.controller_dropdown_menu["values"] = tuple(ControlStrategy.registry.keys())

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
        self.root.exec.world.control_ego_state(cmd=ControlCommand(
            steer=0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_steer_right(self):
        log.debug("Steer right")
        self.root.exec.world.control_ego_state(cmd=ControlCommand(
            steer=-0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def reset_steer(self):
        log.debug("Reset steer")
        self.root.exec.world.control_ego_state(cmd=ControlCommand(
            steer=0), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_acc(self):
        acc = 3
        self.root.exec.world.control_ego_state(
            cmd=ControlCommand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_dec(self):
        acc = -3
        self.root.exec.world.control_ego_state(
            cmd=ControlCommand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

# --------------------------------------------------------------------------------------------
# -Execution----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

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
        self.executer_dropdown_menu["values"] = list(ExecutionStrategy.registry.keys())
        self.executer_dropdown_menu.state(["readonly"])
        self.executer_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.executer_dropdown_menu.pack(side=tk.RIGHT)
        attach_schema_tooltip(self.executer_dropdown_menu, ExecutionSettings, "c40_executer_type")



        ## Second frame
        self.start_exec_button = ttk.Button( exec_second_frame, text="Start", command=self.toggle_exec, style="Start.TButton", width=10,)
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT)
        attach_tooltip(self.start_exec_button, BUTTON_TOOLTIPS["exec_start"])

        btn_stop = ttk.Button( exec_second_frame, text="Stop", command=self.stop_exec, style="Stop.TButton",)
        btn_stop.pack(side=tk.LEFT, padx=1)
        attach_tooltip(btn_stop, BUTTON_TOOLTIPS["exec_stop"])
        btn_step = ttk.Button(exec_second_frame, text="Step", width=4, command=self.step_exec)
        btn_step.pack(side=tk.LEFT)
        attach_tooltip(btn_step, BUTTON_TOOLTIPS["exec_step"])
        btn_reset = ttk.Button(exec_second_frame, text="Reset", width=4, command=self.reset_exec)
        btn_reset.pack(side=tk.LEFT)
        attach_tooltip(btn_reset, BUTTON_TOOLTIPS["exec_reset"])


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


        right_opts = ttk.Frame(exec_second_frame)
        right_opts.pack(side=tk.RIGHT, padx=5, pady=5)

        global_plan_row = ttk.Frame(right_opts)
        global_plan_row.pack(side=tk.TOP, anchor=tk.E)
        global_tj_file = ttk.Entry(
            global_plan_row,
            textvariable=self.root.setting.default_global_plan_file,
            width=15,
            state="readonly",
        )
        global_tj_file.pack(side=tk.RIGHT, padx=(5, 0))
        global_tj_file.bind("<Button-1>", self._pick_default_global_plan)
        lbl = ttk.Label(global_plan_row, text="Default Global Plan")
        lbl.pack(side=tk.RIGHT, padx=(5, 0))
        attach_schema_tooltip(lbl, ExecutionSettings, "c40_global_trajectory")
        attach_schema_tooltip(global_tj_file, ExecutionSettings, "c40_global_trajectory")

        map_row = ttk.Frame(right_opts)
        map_row.pack(side=tk.TOP, anchor=tk.E, pady=(4, 0))
        default_map_entry = ttk.Entry(
            map_row, textvariable=self.root.setting.default_map_file, width=15, state="readonly",
        )
        default_map_entry.pack(side=tk.RIGHT, padx=(5, 0))
        default_map_entry.bind("<Button-1>", self._pick_default_map)
        map_lbl = ttk.Label(map_row, text="Default Map")
        map_lbl.pack(side=tk.RIGHT, padx=(5, 0))
        self._default_map_lbl = map_lbl
        self._default_map_entry = default_map_entry
        self.refresh_default_map_tooltips()


    def refresh_default_map_tooltips(self):
        field = DataPicker.default_map_settings_field()
        update_schema_tooltip(self._default_map_lbl, ExecutionSettings, field)
        update_schema_tooltip(self._default_map_entry, ExecutionSettings, field)


    def _pick_default_global_plan(self, _event=None):
        current = DataPicker.display_path(self.root.setting.default_global_plan_file.get())
        items = DataPicker.list_global_plan_candidates()
        dialog = ThemedListPickerDialog(
            self.root, "Default Global Plan", items, initial=current,
        )
        if dialog.result:
            self.root.setting.default_global_plan_file.set(dialog.result)
            self.root.reload_stack(reload_code=False)

    def _pick_default_map(self, _event=None):
        current = DataPicker.display_path(self.root.setting.default_map_file.get())
        items = DataPicker.list_map_candidates()
        dialog = ThemedListPickerDialog(
            self.root, "Default Map", items, initial=current,
        )
        if dialog.result:
            self.root.setting.default_map_file.set(dialog.result)
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
        if self.root.exec is not None:
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
        """Refresh the executer and bridge dropdowns from the registries."""
        self.executer_dropdown_menu["values"] = list(ExecutionStrategy.registry.keys())
        self.bridge_frame.update_data()

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
        self.world_bridge_dropdown_menu = ttk.Combobox(self, textvariable=self.root.setting.execution_bridge, width=10, state="readonly",)
        self.world_bridge_dropdown_menu["values"] = list(WorldBridge.registry.keys())
        self.world_bridge_dropdown_menu.state(["readonly"])
        self.world_bridge_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.world_bridge_dropdown_menu.grid(row=0, column=0, pady=(0, 4), sticky="we")
        attach_schema_tooltip(self.world_bridge_dropdown_menu, ExecutionSettings, "c40_bridge")

        # Scrollable checklist of the bridge's providable capabilities.
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background")
        self._caps_canvas = tk.Canvas(self, height=96, width=140, highlightthickness=0, bd=0, background=bg_color)
        self._caps_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._caps_canvas.yview)
        self._caps_inner = ttk.Frame(self._caps_canvas)
        self._caps_canvas.configure(yscrollcommand=self._caps_scrollbar.set)
        self._caps_inner.bind(
            "<Configure>", lambda e: self._caps_canvas.configure(scrollregion=self._caps_canvas.bbox("all"))
        )
        self._caps_canvas.create_window((0, 0), window=self._caps_inner, anchor="nw")
        self._caps_canvas.grid(row=1, column=0, sticky="nsew")
        self._caps_scrollbar.grid(row=1, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._cap_vars: dict[str, tk.BooleanVar] = {}
        self._build_capability_checks(set(), set())

    def _build_capability_checks(self, world_capabilities: set, stack_capabilities: set):
        """Rebuild the checklist for the data the selected bridge can feed the stack."""
        for child in self._caps_inner.winfo_children():
            child.destroy()
        self._cap_vars = {}

        # World "action" capabilities are features, not data fed to the stack.
        actions = {WorldCapability.AGENT_SPAWN, WorldCapability.AGENT_CONTROL}
        sensors = sorted(set(world_capabilities) - actions, key=lambda c: c.value)
        ground_truth = sorted(set(stack_capabilities), key=lambda c: c.value)
        for cap in (*sensors, *ground_truth):
            var = tk.BooleanVar(value=is_capability_provided(cap))
            var.trace_add("write", lambda *_: self._on_toggle())
            chk = ttk.Checkbutton(self._caps_inner, text=cap.name, variable=var)
            chk.pack(anchor="w")
            self._cap_vars[cap.name] = var

    def _on_toggle(self):
        """Persist the checked capabilities to the c41_provided filter and refresh plots."""
        ExecutionSettings.c41_provided = [name for name, var in self._cap_vars.items() if var.get()]
        self.root.update_views()

    def update_data(self):
        """Refresh the bridge dropdown from the registry."""
        self.world_bridge_dropdown_menu["values"] = list(WorldBridge.registry.keys())

    def update_for_bridge(self, capabilities: set, stack_capabilities: set | None = None):
        """Rebuild the capability checklist for the active bridge."""
        self._build_capability_checks(capabilities, stack_capabilities or set())




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



