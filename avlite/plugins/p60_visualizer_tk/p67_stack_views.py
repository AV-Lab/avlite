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
from avlite.c10_perception.c14_mapping_strategy import MapReader, MappingStrategy
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
from avlite.c40_execution.c41_world_bridge import (
    WorldBridge,
    is_world_capability_enabled,
    is_world_stack_capability_enabled,
)
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c43_task_strategy import TaskStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c69_settings import AppSettings
from avlite.c60_apps.c65_setting_utils import save_setting
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import (
    ValueGauge,
    DataPicker,
    BUTTON_TOOLTIPS,
    HoverTooltip,
    ThemedListPickerDialog,
    make_strategy_contract_controls,
    make_world_bridge_contract_controls,
    show_strategy_contract_popup,
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

        # Row 0: main perception dropdown + Localization dropdown
        self.perception_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.perception_type, state="readonly", width=14)
        self.perception_dropdown_menu["values"] = ("",) + tuple(PerceptionStrategy.registry.keys())
        self.perception_dropdown_menu.bind("<<ComboboxSelected>>", self._on_perception_selected)
        self.perception_dropdown_menu.grid(row=0, column=0, sticky="ew", padx=2)
        HoverTooltip.attach_schema(self.perception_dropdown_menu, ExecutionSettings, "c40_perception")
        _, perc_info = make_strategy_contract_controls(
            self, self.perception_dropdown_menu, PerceptionStrategy.registry, lambda: self.root.exec
        )
        perc_info.grid(row=0, column=1, padx=(0, 2))

        ttk.Label(self, text="loc:").grid(row=0, column=2, sticky="e", padx=(4, 0))
        self.localization_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.localization_type, state="readonly", width=12)
        self.localization_dropdown_menu["values"] = ("",) + tuple(LocalizationStrategy.registry.keys())
        self.localization_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.localization_dropdown_menu.grid(row=0, column=3, sticky="ew", padx=2)
        HoverTooltip.attach_schema(self.localization_dropdown_menu, ExecutionSettings, "c40_localization")
        _, loc_info = make_strategy_contract_controls(
            self, self.localization_dropdown_menu, LocalizationStrategy.registry, lambda: self.root.exec
        )
        loc_info.grid(row=0, column=4, padx=(0, 2))

        # Rows 1-3: pipeline sub-strategy widgets (shown only for PerceptionPipeline)
        self._lbl_detect = ttk.Label(self, text="Detect:")
        self._lbl_detect.grid(row=1, column=0, sticky="e", padx=(5, 0))
        self.detection_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.detection_strategy_type, state="readonly")
        self.detection_dropdown_menu["values"] = tuple(DetectionStrategy.registry.keys())
        self.detection_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        self.detection_dropdown_menu.grid(row=1, column=1, columnspan=3, sticky="ew")
        _, det_info = make_strategy_contract_controls(
            self, self.detection_dropdown_menu, DetectionStrategy.registry, lambda: self.root.exec
        )
        det_info.grid(row=1, column=4, padx=(0, 2))

        self._lbl_track = ttk.Label(self, text="Track:")
        self._lbl_track.grid(row=2, column=0, sticky="e", padx=(5, 0))
        self.tracking_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.tracking_strategy_type, state="readonly")
        self.tracking_dropdown_menu["values"] = tuple(TrackingStrategy.registry.keys())
        self.tracking_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        self.tracking_dropdown_menu.grid(row=2, column=1, columnspan=3, sticky="ew")
        _, track_info = make_strategy_contract_controls(
            self, self.tracking_dropdown_menu, TrackingStrategy.registry, lambda: self.root.exec
        )
        track_info.grid(row=2, column=4, padx=(0, 2))

        self._lbl_predict = ttk.Label(self, text="Predict:")
        self._lbl_predict.grid(row=3, column=0, sticky="e", padx=(5, 0))
        self.prediction_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.prediction_strategy_type, state="readonly")
        self.prediction_dropdown_menu["values"] = ("",) + tuple(PredictionStrategy.registry.keys())
        self.prediction_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        self.prediction_dropdown_menu.grid(row=3, column=1, columnspan=3, sticky="ew")
        _, pred_info = make_strategy_contract_controls(
            self, self.prediction_dropdown_menu, PredictionStrategy.registry, lambda: self.root.exec
        )
        pred_info.grid(row=3, column=4, padx=(0, 2))
        HoverTooltip.attach_schema(self.detection_dropdown_menu, PerceptionSettings, "c12_detection_strategy")
        HoverTooltip.attach_schema(self.tracking_dropdown_menu, PerceptionSettings, "c12_tracking_strategy")
        HoverTooltip.attach_schema(self.prediction_dropdown_menu, PerceptionSettings, "c12_prediction_strategy")

        # Row 4 (last): mapping strategy + Default Map (map file shown only for MapReader)
        self.mapping_dropdown_menu = ttk.Combobox(
            self, textvariable=self.root.setting.mapping_type, state="readonly", width=14)
        self.mapping_dropdown_menu["values"] = ("",) + tuple(MappingStrategy.registry.keys())
        self.mapping_dropdown_menu.bind("<<ComboboxSelected>>", self._on_mapping_selected)
        self.mapping_dropdown_menu.grid(row=4, column=0, sticky="ew", padx=2)
        HoverTooltip.attach_schema(self.mapping_dropdown_menu, ExecutionSettings, "c40_mapping")
        _, map_info = make_strategy_contract_controls(
            self, self.mapping_dropdown_menu, MappingStrategy.registry, lambda: self.root.exec
        )
        map_info.grid(row=4, column=1, padx=(0, 2))

        self._default_map_lbl = ttk.Label(self, text="Default Map")
        self._default_map_lbl.grid(row=4, column=2, sticky="e", padx=(8, 0))
        self._default_map_entry = ttk.Entry(
            self, textvariable=self.root.setting.default_map_file, width=15, state="readonly",
        )
        self._default_map_entry.grid(row=4, column=3, sticky="ew", padx=2)
        self._default_map_entry.bind("<Button-1>", self._pick_default_map)
        self.refresh_default_map_tooltips()

        self.columnconfigure(0, weight=1)

        self._pipeline_widgets = [
            self._lbl_detect, self.detection_dropdown_menu, det_info,
            self._lbl_track, self.tracking_dropdown_menu, track_info,
            self._lbl_predict, self.prediction_dropdown_menu, pred_info,
        ]
        self._default_map_widgets = [self._default_map_lbl, self._default_map_entry]

        self.root.setting.perception_type.trace_add("write", lambda *_: self._update_pipeline_visibility())
        self.root.setting.mapping_type.trace_add("write", lambda *_: self._update_default_map_visibility())
        self._update_pipeline_visibility()
        self._update_default_map_visibility()

    def _on_perception_selected(self, event=None):
        self._update_pipeline_visibility()
        self.root.reload_stack(reload_code=False)

    def _on_mapping_selected(self, event=None):
        self._update_default_map_visibility()
        self.root.reload_stack(reload_code=False)

    def _update_pipeline_visibility(self):
        is_pipeline = self.root.setting.perception_type.get() == PerceptionPipeline.__name__
        for w in self._pipeline_widgets:
            if is_pipeline:
                w.grid()
            else:
                w.grid_remove()

    def _update_default_map_visibility(self):
        show = self.root.setting.mapping_type.get() == MapReader.__name__
        for w in self._default_map_widgets:
            if show:
                w.grid()
            else:
                w.grid_remove()

    def refresh_default_map_tooltips(self):
        field = DataPicker.default_map_settings_field()
        HoverTooltip.update_schema(self._default_map_lbl, ExecutionSettings, field)
        HoverTooltip.update_schema(self._default_map_entry, ExecutionSettings, field)

    def _pick_default_map(self, _event=None):
        current = DataPicker.display_path(self.root.setting.default_map_file.get())
        items = DataPicker.list_map_candidates()
        dialog = ThemedListPickerDialog(
            self.root, "Default Map", items, initial=current,
        )
        if dialog.result is not None:
            self.root.setting.default_map_file.set(dialog.result)
            self.root.reload_stack(reload_code=False)

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
        log.info("allowed_default_plugins: %s, allowed_community_plugins: %s", allowed_default_plugins, allowed_community_plugins)
        log.info(f"final Strategies: {data}")

        self.perception_dropdown_menu["values"] = ("",) + tuple(data)
        self.localization_dropdown_menu["values"] = ("",) + tuple(LocalizationStrategy.registry.keys())
        self.mapping_dropdown_menu["values"] = ("",) + tuple(MappingStrategy.registry.keys())
        if self.root.exec is None:
            self._update_pipeline_visibility()
            self._update_default_map_visibility()
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
        self._update_default_map_visibility()


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
        self.global_planner_dropdown_menu = ttk.Combobox(global_frame, textvariable=self.root.setting.global_planner_type, width=10)
        self.global_planner_dropdown_menu["values"] = ("",) + tuple(GlobalPlannerStrategy.registry.keys())
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        HoverTooltip.attach_schema(self.global_planner_dropdown_menu, ExecutionSettings, "c40_global_planner")
        _, gp_info = make_strategy_contract_controls(
            global_frame, self.global_planner_dropdown_menu, GlobalPlannerStrategy.registry, lambda: self.root.exec
        )
        gp_info.pack(side=tk.LEFT)

        btn_global_replan = ttk.Button(global_frame, text="Global Replan", command=self.replan_global)
        btn_global_replan.pack(side=tk.LEFT, fill=tk.X, expand=True)
        HoverTooltip.attach(btn_global_replan, BUTTON_TOOLTIPS["plan_global_replan"])
        btn_save_global = ttk.Button(global_frame, text="⬇", command=self.save_global_plan, width=3)
        btn_save_global.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_save_global, BUTTON_TOOLTIPS["plan_save_global"])

        global_plan_row = ttk.Frame(self)
        global_plan_row.pack(fill=tk.X)
        lbl = ttk.Label(global_plan_row, text="Default Global Plan")
        lbl.pack(side=tk.LEFT, padx=(5, 0))
        global_tj_file = ttk.Entry(
            global_plan_row,
            textvariable=self.root.setting.default_global_plan_file,
            width=15,
            state="readonly",
        )
        global_tj_file.pack(side=tk.LEFT,fill=tk.X, expand=True, padx=(5, 0))
        global_tj_file.bind("<Button-1>", self._pick_default_global_plan)
        HoverTooltip.attach_schema(lbl, ExecutionSettings, "c40_global_trajectory")
        HoverTooltip.attach_schema(global_tj_file, ExecutionSettings, "c40_global_trajectory")

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=2)

        # - Local -----
        wp_frame = ttk.Frame(self)
        wp_frame.pack(fill=tk.X)

        self.local_planner_dropdown_menu = ttk.Combobox(wp_frame, textvariable=self.root.setting.local_planner_type, width=10)
        self.local_planner_dropdown_menu["values"] = ("",) + tuple(LocalPlanningStrategy.registry.keys())
        self.local_planner_dropdown_menu.state(["readonly"])
        self.local_planner_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.local_planner_dropdown_menu.bind("<<ComboboxSelected>>", self._on_local_planner_selected)
        HoverTooltip.attach_schema(self.local_planner_dropdown_menu, ExecutionSettings, "c40_local_planner")
        _, lp_info = make_strategy_contract_controls(
            wp_frame, self.local_planner_dropdown_menu, LocalPlanningStrategy.registry, lambda: self.root.exec
        )
        lp_info.pack(side=tk.LEFT)

        btn_set_wp = ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint)
        btn_set_wp.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_set_wp, BUTTON_TOOLTIPS["plan_set_waypoint"])
        global_tj_wp_entry = ttk.Entry( wp_frame, width=6, textvariable=self.root.setting.current_wp)
        global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        global_tj_wp_entry.bind("<Return>", self.text_on_enter)
        self._wp_count_label = ttk.Label(wp_frame, text="0")
        self._wp_count_label.pack(side=tk.LEFT, padx=5)

        self._lap_label = ttk.Label(self, text="Lap: ")
        self._local_sub_frame = ttk.Frame(self)
        self._local_sub_frame.pack(fill=tk.X)

        self._lbl_behavior = ttk.Label(self._local_sub_frame, text="Behavior:")
        self.behavioral_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.behavioral_strategy_type, state="readonly", width=8)
        self.behavioral_dropdown_menu["values"] = ("",) + tuple(LocalBehavioralPlanningStrategy.registry.keys())
        self.behavioral_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        HoverTooltip.attach_schema(self.behavioral_dropdown_menu, PlanningSettings, "c23_behavioral_strategy")
        _, beh_info = make_strategy_contract_controls(
            self._local_sub_frame, self.behavioral_dropdown_menu,
            LocalBehavioralPlanningStrategy.registry, lambda: self.root.exec,
        )
        self._lbl_path = ttk.Label(self._local_sub_frame, text="Path:")
        self.path_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.path_strategy_type, state="readonly", width=8)
        self.path_dropdown_menu["values"] = ("",) + tuple(LocalPathPlanningStrategy.registry.keys())
        self.path_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        HoverTooltip.attach_schema(self.path_dropdown_menu, PlanningSettings, "c23_path_strategy")
        _, path_info = make_strategy_contract_controls(
            self._local_sub_frame, self.path_dropdown_menu,
            LocalPathPlanningStrategy.registry, lambda: self.root.exec,
        )
        self._lbl_speed = ttk.Label(self._local_sub_frame, text="Speed:")
        self.velocity_dropdown_menu = ttk.Combobox(
            self._local_sub_frame, textvariable=self.root.setting.velocity_strategy_type, state="readonly", width=8)
        self.velocity_dropdown_menu["values"] = ("",) + tuple(LocalVelocityPlanningStrategy.registry.keys())
        self.velocity_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.refresh_pipeline())
        HoverTooltip.attach_schema(self.velocity_dropdown_menu, PlanningSettings, "c23_velocity_strategy")
        _, vel_info = make_strategy_contract_controls(
            self._local_sub_frame, self.velocity_dropdown_menu,
            LocalVelocityPlanningStrategy.registry, lambda: self.root.exec,
        )
        self._pipeline_widgets = (
            (self._lbl_behavior, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.behavioral_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (beh_info, {"side": tk.LEFT}),
            (self._lbl_path, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.path_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (path_info, {"side": tk.LEFT}),
            (self._lbl_speed, {"side": tk.LEFT, "padx": (5, 0)}),
            (self.velocity_dropdown_menu, {"side": tk.LEFT, "fill": tk.X, "expand": True}),
            (vel_info, {"side": tk.LEFT}),
        )

        self.root.setting.local_planner_type.trace_add("write", lambda *_: self._update_pipeline_visibility())
        self._update_pipeline_visibility()

        self._lap_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(self, font=self.root.small_font,
                  textvariable=self.root.setting.lap).pack(side=tk.LEFT, padx=5)

        btn_wp_back = ttk.Button(self, text="◀️", command=self.step_waypoint_back, width=2)
        btn_wp_back.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_wp_back, BUTTON_TOOLTIPS["plan_wp_back"])
        btn_plan_step = ttk.Button(self, text="▶", command=self.step_plan, width=2)
        btn_plan_step.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_plan_step, BUTTON_TOOLTIPS["plan_step"])
        btn_plan_align = ttk.Button(self, text="Align", command=self.align_plan, width=4)
        btn_plan_align.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_plan_align, BUTTON_TOOLTIPS["plan_align"])
        btn_local_replan = ttk.Button(self, text="Local Replan", command=self.replan)
        btn_local_replan.pack(side=tk.LEFT, fill=tk.X, expand=True)
        HoverTooltip.attach(btn_local_replan, BUTTON_TOOLTIPS["plan_local_replan"])

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

    def _pick_default_global_plan(self, _event=None):
        current = DataPicker.display_path(self.root.setting.default_global_plan_file.get())
        items = DataPicker.list_global_plan_candidates()
        dialog = ThemedListPickerDialog(
            self.root, "Default Global Plan", items, initial=current,
        )
        if dialog.result is not None:
            self.root.setting.default_global_plan_file.set(dialog.result)
            self.root.reload_stack(reload_code=False)

    def update_data(self):
        """Update data in the plan frame."""
        self.local_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.local_planner_dropdown_menu["values"] = ("",) + tuple(LocalPlanningStrategy.registry.keys())
        self.global_planner_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.global_planner_dropdown_menu["values"] = ("",) + tuple(GlobalPlannerStrategy.registry.keys())
        self.behavioral_dropdown_menu["values"] = ("",) + tuple(LocalBehavioralPlanningStrategy.registry.keys())
        self.path_dropdown_menu["values"] = ("",) + tuple(LocalPathPlanningStrategy.registry.keys())
        self.velocity_dropdown_menu["values"] = ("",) + tuple(LocalVelocityPlanningStrategy.registry.keys())
        self._update_pipeline_visibility()
        if self.root.exec is not None and self.root.exec.local_planner is not None:
            self._wp_count_label.config(
                text=f"{len(self.root.exec.local_planner.global_trajectory.path_x) - 1}"
            )
        else:
            self._wp_count_label.config(text="0")

    def set_waypoint(self):
        if not self.root.exec or not self.root.exec.local_planner:
            return
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()
    def step_waypoint_back(self):
        """ Step back to the previous waypoint in the local planner."""
        if not self.root.exec or not self.root.exec.local_planner:
            return
        self.root.setting.current_wp.set(str(int(self.root.setting.current_wp.get()) - 1))
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()
    
    def text_on_enter(self, event):
        widget = event.widget  # Get the widget that triggered the event
        text = widget.get()    # Retrieve the text from the widget
        self.root.validate_float_input(text)  # Validate the input
        log.debug("Text entered: %s", text)
        widget.tk_focusNext().focus_set()  # Move focus to the next widget
        if not self.root.exec or not self.root.exec.local_planner:
            return
        self.root.exec.local_planner.reset(wp=int(self.root.setting.current_wp.get()))
        self.root.update_ui()

    def replan(self):
        if not self.root.exec or not self.root.exec.local_planner:
            return
        t1 = time.time()
        self.root.exec.local_planner.replan(
            perception_model=self.root.exec.pm,
            sensors=self.root.exec.world.get_sensor_frame(),
        )
        t2 = time.time()
        log.info(f"Re-plan Time: {(t2-t1)*1000:.2f} ms")
        self.root.update_ui()

    def replan_global(self):
        if not self.root.exec or not self.root.exec.global_planner:
            return
        self.root.replan_global()
        self.root.local_plan_plot_view.reset()
        self.root.global_plan_plot_view.plot()
        self.root.local_plan_plot_view.plot()
        self.root.update_ui()

    def save_global_plan(self):
        if not self.root.exec or not self.root.exec.global_planner:
            return
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
        if not self.root.exec or not self.root.exec.local_planner:
            return
        log.debug("Aligning plan with current ego state")
        self.root.exec.local_planner.step(self.root.exec.world.get_ego_state())
        self.root.update_ui()

    def step_plan(self):
        if not self.root.exec or not self.root.exec.local_planner:
            return
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
        control_button_frame.pack(fill=tk.X)
        self.controller_dropdown_menu = ttk.Combobox(control_button_frame, textvariable=self.root.setting.controller_type, width=10)
        self.controller_dropdown_menu["values"] = ("",) + tuple(ControlStrategy.registry.keys())
        self.controller_dropdown_menu.state(["readonly"])
        self.controller_dropdown_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.controller_dropdown_menu.bind("<<ComboboxSelected>>", lambda event: self.root.reload_stack(reload_code=False))
        HoverTooltip.attach_schema(self.controller_dropdown_menu, ExecutionSettings, "c40_controller")
        _, cn_info = make_strategy_contract_controls(
            control_button_frame, self.controller_dropdown_menu, ControlStrategy.registry, lambda: self.root.exec
        )
        cn_info.pack(side=tk.LEFT)

        btn_control_step = ttk.Button(control_button_frame, text="Step", command=self.step_control)
        btn_control_step.pack(side=tk.LEFT, fill=tk.X, expand=True)
        HoverTooltip.attach(btn_control_step, BUTTON_TOOLTIPS["control_step"])
        btn_control_align = ttk.Button(control_button_frame, text="Align", width=4, command=self.align_control)
        btn_control_align.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_control_align, BUTTON_TOOLTIPS["control_align"])
        btn_steer_left = ttk.Button(control_button_frame, text="◀️ ", width=2, command=self.step_steer_left)
        btn_steer_left.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_steer_left, BUTTON_TOOLTIPS["control_steer_left"])
        btn_steer_right = ttk.Button(control_button_frame, text="▶", width=2, command=self.step_steer_right)
        btn_steer_right.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_steer_right, BUTTON_TOOLTIPS["control_steer_right"])
        btn_accel = ttk.Button(control_button_frame, text="▲", width=2, command=self.step_acc)
        btn_accel.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_accel, BUTTON_TOOLTIPS["control_accel"])
        btn_decel = ttk.Button(control_button_frame, text="▼", width=2, command=self.step_dec)
        btn_decel.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_decel, BUTTON_TOOLTIPS["control_decel"])

        #################
        # Progress bars
        #################
        # Each label sits on the same grid row as its gauge so they stay aligned.
        self.cte_frame = ttk.Frame(self)
        self.cte_frame.pack(fill=tk.X)
        self.cte_frame.columnconfigure(1, weight=1)

        ttk.Label(self.cte_frame, text="Vel CTE", font=self.root.small_font).grid(row=0, column=0, sticky="w", padx=5)
        self.gauge_cte_vel = ValueGauge(self.cte_frame, min_value=-20, max_value=20, dpi_scale=self.root._dpi_scale)
        self.gauge_cte_vel.grid(row=0, column=1, sticky="ew", pady=1)

        ttk.Label(self.cte_frame, text="Pos CTE", font=self.root.small_font).grid(row=1, column=0, sticky="w", padx=5)
        self.gauge_cte_steer = ValueGauge(self.cte_frame, min_value=-20, max_value=20, dpi_scale=self.root._dpi_scale)
        self.gauge_cte_steer.grid(row=1, column=1, sticky="ew", pady=1)

        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X)
        self.progress_frame.columnconfigure(1, weight=1)

        ttk.Label(self.progress_frame, text="Accel", font=self.root.small_font).grid(row=0, column=0, sticky="w", padx=5)
        self.gauge_acc = ValueGauge(
            self.progress_frame,
            min_value=ControlSettings.c32_ego_min_acceleration,
            max_value=ControlSettings.c32_ego_max_acceleration,
            dpi_scale=self.root._dpi_scale,
        )
        self.gauge_acc.grid(row=0, column=1, sticky="ew", pady=1)

        ttk.Label(self.progress_frame, text="Steer", font=self.root.small_font).grid(row=1, column=0, sticky="w", padx=5)
        self.gauge_steer = ValueGauge(
            self.progress_frame,
            min_value=ControlSettings.c32_ego_min_steering,
            max_value=ControlSettings.c32_ego_max_steering,
            dpi_scale=self.root._dpi_scale,
        )
        self.gauge_steer.grid(row=1, column=1, sticky="ew", pady=1)
        # ----

    def update_data(self):
        """Update data in the control frame."""
        self.controller_dropdown_menu.delete(0, tk.END)  # Clear existing values
        self.controller_dropdown_menu["values"] = ("",) + tuple(ControlStrategy.registry.keys())

    def step_control(self):
        if not self.root.exec or not self.root.exec.controller or not self.root.exec.local_planner:
            return
        cmd = self.root.exec.controller.control(
            self.root.exec.ego_state,
            self.root.exec.local_planner.get_local_plan(),
            control_dt=self.root.setting.sim_dt.get(),
            perception_model=self.root.exec.pm,
            sensors=self.root.exec.world.get_sensor_frame(),
        )

        self.root.apply_world_control(cmd, dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def align_control(self):
        """Snap plant + stack ego to the plan location (same dual-write as teleport)."""
        if not self.root.exec or not self.root.exec.controller or not self.root.exec.local_planner:
            return
        x, y = self.root.exec.local_planner.location_xy
        # Must move world ego and sync stack PM — mutating only exec.ego_state is undone
        # on the next GT-localization tick after the world/stack ego split.
        self.root.teleport_ego(x, y)
        self.root.exec.controller.reset()
        self.root.update_ui()

    def step_steer_left(self):
        log.debug("Steer right")
        self.root.apply_world_control(
            ControlCommand(steer=0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_steer_right(self):
        log.debug("Steer right")
        self.root.apply_world_control(
            ControlCommand(steer=-0.7), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def reset_steer(self):
        log.debug("Reset steer")
        self.root.apply_world_control(
            ControlCommand(steer=0), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_acc(self):
        acc = 3
        self.root.apply_world_control(
            ControlCommand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

    def step_dec(self):
        acc = -3
        self.root.apply_world_control(
            ControlCommand(acceleration=acc), dt=self.root.setting.sim_dt.get())
        self.root.update_ui()

# --------------------------------------------------------------------------------------------
# -Execution----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

class ExecView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)

        self.root = root
        self._exec_after_id: str | None = None

        # Side-by-side panes share height (same pattern as PerceivePlanControlView).
        exec_bar = ttk.Frame(self)
        exec_bar.pack(fill=tk.X)

        self.execution_factory_frame = ttk.LabelFrame(exec_bar, text="Execution")
        self.execution_factory_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 2))

        executer_frame = ExecSettingsFrame(self.root, exec_bar)
        executer_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        self.bridge_frame = BridgeFrame(self.root, exec_bar)
        self.bridge_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        exec_stats_frame = ExecStatsFrame(self.root, exec_bar)
        exec_stats_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(2, 0))

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
        chk_pace_pr = ttk.Checkbutton(
            exec_first_frame,
            variable=self.root.setting.pace_perception,
            width=0,
            command=self._update_pace_entry_states,
        )
        chk_pace_pr.pack(side=tk.LEFT, padx=(5, 0), pady=1)
        lbl = ttk.Label(exec_first_frame, text="Perception \u0394t ")
        lbl.pack(side=tk.LEFT, pady=1)
        self.dt_perception_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.perception_dt, width=5,)
        self.dt_perception_entry.pack(side=tk.LEFT)
        self.dt_perception_entry.bind("<Return>", self.text_on_enter)
        HoverTooltip.attach_schema(lbl, ExecutionSettings, "c40_perception_dt")
        HoverTooltip.attach_schema(self.dt_perception_entry, ExecutionSettings, "c40_perception_dt")
        HoverTooltip.attach_schema(chk_pace_pr, ExecutionSettings, "c40_pace_perception")

        chk_pace_pl = ttk.Checkbutton(
            exec_first_frame,
            variable=self.root.setting.pace_replan,
            width=0,
            command=self._update_pace_entry_states,
        )
        chk_pace_pl.pack(side=tk.LEFT, padx=(5, 0), pady=1)
        lbl = ttk.Label(exec_first_frame, text="Replan Δt ")
        lbl.pack(side=tk.LEFT, pady=1)
        self.dt_plan_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.replan_dt, width=5,)
        self.dt_plan_entry.pack(side=tk.LEFT)
        self.dt_plan_entry.bind("<Return>", self.text_on_enter)
        HoverTooltip.attach_schema(lbl, ExecutionSettings, "c40_replan_dt")
        HoverTooltip.attach_schema(self.dt_plan_entry, ExecutionSettings, "c40_replan_dt")
        HoverTooltip.attach_schema(chk_pace_pl, ExecutionSettings, "c40_pace_replan")

        chk_pace_cn = ttk.Checkbutton(
            exec_first_frame,
            variable=self.root.setting.pace_control,
            width=0,
            command=self._update_pace_entry_states,
        )
        chk_pace_cn.pack(side=tk.LEFT, padx=(5, 0), pady=1)
        lbl = ttk.Label(exec_first_frame, text="Control Δt ")
        lbl.pack(side=tk.LEFT, pady=1)
        self.dt_control_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.control_dt, width=5,)
        self.dt_control_entry.pack(side=tk.LEFT)
        self.dt_control_entry.bind("<Return>", self.text_on_enter)
        HoverTooltip.attach_schema(lbl, ExecutionSettings, "c40_control_dt")
        HoverTooltip.attach_schema(self.dt_control_entry, ExecutionSettings, "c40_control_dt")
        HoverTooltip.attach_schema(chk_pace_cn, ExecutionSettings, "c40_pace_control")

        chk_pace_sim = ttk.Checkbutton(
            exec_first_frame,
            variable=self.root.setting.pace_sim,
            width=0,
            command=self._update_pace_entry_states,
        )
        chk_pace_sim.pack(side=tk.LEFT, padx=(5, 0), pady=1)
        lbl = ttk.Label(exec_first_frame, text="Sim Δt ")
        lbl.pack(side=tk.LEFT, pady=1)
        self.sim_dt_entry = ttk.Entry(exec_first_frame, textvariable=self.root.setting.sim_dt, width=5,)
        self.sim_dt_entry.pack(side=tk.LEFT)
        self.sim_dt_entry.bind("<Return>", self.text_on_enter)
        HoverTooltip.attach_schema(lbl, ExecutionSettings, "c40_sim_dt")
        HoverTooltip.attach_schema(self.sim_dt_entry, ExecutionSettings, "c40_sim_dt")
        HoverTooltip.attach_schema(chk_pace_sim, ExecutionSettings, "c40_pace_sim")

        for pace_var in (
            self.root.setting.pace_perception,
            self.root.setting.pace_replan,
            self.root.setting.pace_control,
            self.root.setting.pace_sim,
        ):
            pace_var.trace_add("write", self._update_pace_entry_states)
        self._update_pace_entry_states()

        ## Second frame
        self.start_exec_button = ttk.Button( exec_second_frame, text="Start", command=self.toggle_exec, style="Start.TButton", width=10,)
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT, padx = (5,0))
        HoverTooltip.attach(self.start_exec_button, BUTTON_TOOLTIPS["exec_start"])

        btn_stop = ttk.Button( exec_second_frame, text="Stop", command=self.stop_exec, style="Stop.TButton",)
        btn_stop.pack(side=tk.LEFT, padx=1)
        HoverTooltip.attach(btn_stop, BUTTON_TOOLTIPS["exec_stop"])
        btn_step = ttk.Button(exec_second_frame, text="Step", width=4, command=self.step_exec)
        btn_step.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_step, BUTTON_TOOLTIPS["exec_step"])
        btn_reset = ttk.Button(exec_second_frame, text="Reset", width=4, command=self.reset_exec)
        btn_reset.pack(side=tk.LEFT)
        HoverTooltip.attach(btn_reset, BUTTON_TOOLTIPS["exec_reset"])
        
        self.executer_dropdown_menu = ttk.Combobox(exec_second_frame, textvariable=self.root.setting.executer_type, state="readonly",)
        self.executer_dropdown_menu["values"] = list(ExecutionStrategy.registry.keys())
        self.executer_dropdown_menu.state(["readonly"])
        self.executer_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.executer_dropdown_menu.pack(side=tk.RIGHT)
        HoverTooltip.attach_schema(self.executer_dropdown_menu, ExecutionSettings, "c40_executer_type")
        
        ttk.Label(exec_second_frame, text="Executer: ").pack(side=tk.RIGHT, padx=5)


        ## Third frame — Tasks label, wrapping chips, and + on one row
        tasks_row = ttk.Frame(exec_third_frame)
        tasks_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)
        lbl_tasks = ttk.Label(tasks_row, text="Tasks:")
        lbl_tasks.pack(side=tk.LEFT)
        HoverTooltip.attach_schema(lbl_tasks, ExecutionSettings, "c40_execution_tasks")
        self._tasks_add_btn = ttk.Button(tasks_row, text="+", width=2, command=self._add_execution_task)
        self._tasks_add_btn.pack(side=tk.RIGHT, padx=(2, 0))
        HoverTooltip.attach(
            self._tasks_add_btn,
            "Add a TaskStrategy from the registry (append). Reload stack after change.",
        )
        self._tasks_chips = ttk.Frame(tasks_row)
        self._tasks_chips.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self._task_chip_widgets: list = []
        self._tasks_chips_last_width = None
        self._tasks_chips.bind("<Configure>", self._rebuild_task_chips)
        self.root.setting.execution_tasks.trace_add("write", lambda *_: self._rebuild_task_chips())
        self._rebuild_task_chips()

        state_row = ttk.Frame(exec_third_frame)
        state_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)
        btn_set_start = ttk.Button(state_row, text="Save Start", width=10, command=self.set_start)
        btn_set_start.pack(side=tk.RIGHT, padx=(2, 0))
        HoverTooltip.attach(btn_set_start, BUTTON_TOOLTIPS["exec_set_start"])
        vehicle_state_label = ttk.Label(state_row, font=self.root.small_font, textvariable=self.root.setting.vehicle_state)
        vehicle_state_label.pack(side=tk.LEFT, fill=tk.X, expand=True)


    def _rebuild_task_chips(self, event=None) -> None:
        if event is not None and event.widget is not self._tasks_chips:
            return
        if event is None:
            for child in self._tasks_chips.winfo_children():
                child.destroy()
            self._task_chip_widgets = []
            for name in self.root.setting.execution_task_names():
                chip = ttk.Frame(self._tasks_chips)
                ttk.Label(chip, text=name, font=self.root.small_font).pack(side=tk.LEFT)
                info_btn = ttk.Button(chip, text="ⓘ", width=2)
                info_btn.configure(
                    command=lambda n=name, btn=info_btn: show_strategy_contract_popup(
                        btn,
                        name=n,
                        registry=TaskStrategy.registry,
                        get_exec=lambda: self.root.exec,
                    )
                )
                info_btn.pack(side=tk.LEFT)
                HoverTooltip.attach(
                    info_btn,
                    "World requirements, stack requirements & capabilities",
                )
                ttk.Button(
                    chip,
                    text="\u00d7",
                    width=2,
                    command=lambda n=name: self._remove_execution_task(n),
                ).pack(side=tk.LEFT)
                self._task_chip_widgets.append(chip)

        if event is not None:
            width = max(int(event.width), 1)
        else:
            self._tasks_chips.update_idletasks()
            width = max(int(self._tasks_chips.winfo_width()), 1)
        if width == self._tasks_chips_last_width and event is not None:
            return
        self._tasks_chips_last_width = width

        row = col = used = 0
        for chip in self._task_chip_widgets:
            chip.update_idletasks()
            need = int(chip.winfo_reqwidth()) + 4
            if col and used + need > width:
                row += 1
                col = 0
                used = 0
            chip.grid(row=row, column=col, sticky="w", padx=(0, 4), pady=1)
            used += need
            col += 1

    def _set_execution_task_names(self, names: list[str], *, reload: bool) -> None:
        self.root.setting.execution_tasks.set(",".join(names))
        if reload:
            self.root.reload_stack(reload_code=False)

    def _add_execution_task(self) -> None:
        selected = set(self.root.setting.execution_task_names())
        available = sorted(n for n in TaskStrategy.registry if n not in selected)
        if not available:
            messagebox.showinfo("Add Task", "All registered tasks are already selected.")
            return
        dialog = ThemedListPickerDialog(self.root, "Add Task", available)
        if dialog.result:
            names = self.root.setting.execution_task_names()
            names.append(dialog.result)
            self._set_execution_task_names(names, reload=True)

    def _remove_execution_task(self, name: str) -> None:
        names = [n for n in self.root.setting.execution_task_names() if n != name]
        self._set_execution_task_names(names, reload=True)

    def _update_pace_entry_states(self, *_):
        s = self.root.setting
        pairs = (
            (s.pace_perception, self.dt_perception_entry),
            (s.pace_replan, self.dt_plan_entry),
            (s.pace_control, self.dt_control_entry),
            (s.pace_sim, self.sim_dt_entry),
        )
        for var, entry in pairs:
            entry.state(["!disabled"] if var.get() else ["disabled"])

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
        # Cooperative stop (StopExecAtGoalTask) leaves executer.stopped set; clear
        # it so AsyncThreadedExecuter.step may create workers again on Start.
        if self.root.exec is not None:
            self.root.exec.stopped = False
        # self.start_exec_button.config(state=tk.DISABLED)
        self.start_exec_button.state(['disabled'])
        self.root.update_ui()
        self._exec_loop()

    def _exec_loop(self):
        if self.root.setting.exec_running:
            # Task-driven stop (e.g. StopExecAtGoalTask) flips executer.stopped but
            # not exec_running. Mirror it into the UI so we do not keep polling
            # step() — async step used to recreate workers and clear stopped.
            if self.root.exec is not None and self.root.exec.stopped:
                self.stop_exec()
                return

            current_time = time.time()
            cn_dt = float(self.root.setting.control_dt.get())
            pl_dt = float(self.root.setting.replan_dt.get())
            pr_dt = float(self.root.setting.perception_dt.get())
            sim_dt = float(self.root.setting.sim_dt.get())

            pace_sim = bool(self.root.setting.pace_sim.get())
            self.root.exec.step(
                control_dt=cn_dt,
                replan_dt=pl_dt,
                perception_dt=pr_dt,
                sim_dt=sim_dt,
                call_replan=self.root.setting.exec_plan.get(),
                call_control=self.root.setting.exec_control.get(),
                call_perceive=self.root.setting.exec_perceive.get(),
                call_localize=self.root.setting.exec_localize.get(),
                pace_perception=bool(self.root.setting.pace_perception.get()),
                pace_replan=bool(self.root.setting.pace_replan.get()),
                pace_control=bool(self.root.setting.pace_control.get()),
                pace_sim=pace_sim,
            )

            if self.root.exec.stopped:
                self.stop_exec()
                return

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
            elif pace_sim:
                next_frame_delay = max(0.001, sim_dt - processing_time)
            else:
                next_frame_delay = 0.001
            self._exec_after_id = self.root.after(
                int(next_frame_delay * 1000), self._exec_loop
            )

    def stop_exec(self):
        self.root.setting.exec_running = False
        if self._exec_after_id is not None:
            try:
                self.root.after_cancel(self._exec_after_id)
            except tk.TclError:
                pass
            self._exec_after_id = None
        if self.root.exec is not None:
            self.root.exec.stop()
        # self.start_exec_button.config(state=tk.NORMAL)
        self.start_exec_button.state(['!disabled'])
        self.root.update_ui()

    def step_exec(self):
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
            pace_perception=bool(self.root.setting.pace_perception.get()),
            pace_replan=bool(self.root.setting.pace_replan.get()),
            pace_control=bool(self.root.setting.pace_control.get()),
            pace_sim=bool(self.root.setting.pace_sim.get()),
        )
        self.root.update_ui()

    def update_data(self):
        """Refresh the executer and bridge dropdowns from the registries."""
        self.executer_dropdown_menu["values"] = list(ExecutionStrategy.registry.keys())
        self.bridge_frame.update_data()

    def reset_exec(self):
        self.root.exec.reset()
        self.root.update_ui()

    def set_start(self):
        ego = self.root.exec.world.get_ego_state()
        stack = self.root.exec.ego_state
        ExecutionSettings.c40_start_pose = [ego.x, ego.y, ego.theta]
        # Profile YAML stores pose only; snapshot velocity at 0 so Reset matches a
        # cold start (NPC spawn still captures non-zero velocity via set_start).
        world_v, stack_v = ego.velocity, stack.velocity
        ego.velocity = 0.0
        stack.velocity = 0.0
        ego.set_start()
        stack.set_start()
        ego.velocity = world_v
        stack.velocity = stack_v
        profile = self.root.setting.c60_selected_profile.get()
        save_setting(ExecutionSettings, profile=profile)
        log.info(f"Start pose saved to profile {profile!r}: ({ego.x:.2f}, {ego.y:.2f}, {ego.theta:.2f})")

class ExecSettingsFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Executables")
        self.root = root
        chk = ttk.Checkbutton(self, text="Perception", variable=self.root.setting.exec_perceive)
        chk.grid(row=0, column=0, sticky="w")
        HoverTooltip.attach_schema(chk, VisualizationSettings, "exec_perceive")
        chk = ttk.Checkbutton(self, text="Planning", variable=self.root.setting.exec_plan)
        chk.grid(row=1, column=0, sticky="w")
        HoverTooltip.attach_schema(chk, VisualizationSettings, "exec_plan")
        
        chk = ttk.Checkbutton(self, text="Control", variable=self.root.setting.exec_control)
        chk.grid(row=2, column=0, sticky="w")
        HoverTooltip.attach_schema(chk, VisualizationSettings, "exec_control")

        chk = ttk.Checkbutton(self, text="Localization", variable=self.root.setting.exec_localize)
        chk.grid(row=3, column=0, sticky="w")
        HoverTooltip.attach_schema(chk, VisualizationSettings, "exec_localize")


class BridgeFrame(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp, view):
        super().__init__(view, text="Bridge Setting")
        self.root = root
        self.world_bridge_dropdown_menu = ttk.Combobox(self, textvariable=self.root.setting.execution_bridge, width=10, state="readonly",)
        self.world_bridge_dropdown_menu["values"] = list(WorldBridge.registry.keys())
        self.world_bridge_dropdown_menu.state(["readonly"])
        self.world_bridge_dropdown_menu.bind("<<ComboboxSelected>>", lambda e: self.root.reload_stack(reload_code=False))
        self.world_bridge_dropdown_menu.grid(row=0, column=0, columnspan=2, pady=(0, 2), sticky="we")
        HoverTooltip.attach_schema(self.world_bridge_dropdown_menu, ExecutionSettings, "c40_bridge")
        _, bridge_info = make_world_bridge_contract_controls(
            self, self.world_bridge_dropdown_menu, lambda: self.root.exec
        )
        bridge_info.grid(row=0, column=2, pady=(0, 2), padx=(2, 0), sticky="e")

        ttk.Label(self, text="world capabilities", font=self.root.small_font).grid(row=1, column=0, sticky="w")
        ttk.Label(self, text="stack capabilities", font=self.root.small_font).grid(row=1, column=1, sticky="w")
        self._world_inner = ttk.Frame(self)
        self._world_inner.grid(row=2, column=0, sticky="nw")
        self._stack_inner = ttk.Frame(self)
        self._stack_inner.grid(row=2, column=1, sticky="nw", padx=(8, 0))
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._world_cap_vars: dict[str, tk.BooleanVar] = {}
        self._stack_cap_vars: dict[str, tk.BooleanVar] = {}

    def _build_capability_checks(self, world_capabilities: set, stack_capabilities: set):
        """Rebuild the checklists for the data the selected bridge can feed the stack."""
        for child in (*self._world_inner.winfo_children(), *self._stack_inner.winfo_children()):
            child.destroy()
        self._world_cap_vars = {}
        self._stack_cap_vars = {}

        # World "action" capabilities are features, not data fed to the stack.
        actions = {WorldCapability.AGENT_SPAWN, WorldCapability.AGENT_CONTROL}
        sensors = sorted(set(world_capabilities) - actions, key=lambda c: c.value)
        ground_truth = sorted(set(stack_capabilities), key=lambda c: c.value)
        for cap in sensors:
            var = tk.BooleanVar(value=is_world_capability_enabled(cap))
            var.trace_add("write", lambda *_: self._on_toggle())
            chk = ttk.Checkbutton(self._world_inner, text=cap.name, variable=var)
            chk.pack(anchor="w")
            HoverTooltip.attach_capability(chk, cap)
            self._world_cap_vars[cap.name] = var
        for cap in ground_truth:
            var = tk.BooleanVar(value=is_world_stack_capability_enabled(cap))
            var.trace_add("write", lambda *_: self._on_toggle())
            chk = ttk.Checkbutton(self._stack_inner, text=cap.name, variable=var)
            chk.pack(anchor="w")
            HoverTooltip.attach_capability(chk, cap)
            self._stack_cap_vars[cap.name] = var

    def _on_toggle(self):
        """Persist the checked capabilities to the two c41_world_* filters and refresh plots."""
        ExecutionSettings.c41_world_capabilities = [
            name for name, var in self._world_cap_vars.items() if var.get()
        ]
        ExecutionSettings.c41_world_stack_capabilities = [
            name for name, var in self._stack_cap_vars.items() if var.get()
        ]
        self.root.update_views()

    def update_data(self):
        """Refresh the bridge dropdown from the registry."""
        self.world_bridge_dropdown_menu["values"] = list(WorldBridge.registry.keys())

    def update_for_bridge(self, capabilities: set, stack_capabilities: set | None = None):
        """Rebuild the capability checklist for the active bridge."""
        self._build_capability_checks(capabilities, stack_capabilities or set())

    def update_canvas_theme(self):
        """No-op: capability lists are ttk widgets (theme follows Style)."""
        pass




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



