"""Runtime settings mirror for the Qt visualizer (tk.Variable-free)."""

from __future__ import annotations

from typing import Any, Callable

from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PerceptionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import StackSettingsSync
from avlite.c60_apps.c69_settings import AppSettings

from .settings import PluginSettings


class Var:
    """Minimal get/set/bind stand-in for tk Variable APIs used by the Tk app."""

    __slots__ = ("_value", "_callbacks")

    def __init__(self, value: Any = None):
        self._value = value
        self._callbacks: list[Callable[[], None]] = []

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        if self._value == value:
            return
        self._value = value
        for cb in list(self._callbacks):
            cb()

    def bind(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)


def _strategy_default(registry: dict, configured: str | None) -> str:
    if configured:
        return configured
    keys = list(registry.keys()) if registry else []
    return keys[0] if keys else ""


class VisualizationSettings:
    """Runtime values for the Qt visualizer; synced to stack singletons."""

    schema = PluginSettings

    def __init__(self) -> None:
        self._syncing_stack = False
        self._syncing_perception_pipeline = False
        self._syncing_local_planning_pipeline = False

        self.c62_load_plugins = Var(bool(AppSettings.c62_load_plugins))
        self.c60_selected_profile = Var(AppSettings.c60_selected_profile or "default")
        self.p60_shortcut_mode = Var(bool(PluginSettings.p60_shortcut_mode))
        self.p60_next_profile = Var(PluginSettings.p60_next_profile)
        self.p60_dark_mode = Var(bool(PluginSettings.p60_dark_mode))
        self.p60_bg_color = PluginSettings.p60_bg_color
        self.p60_fg_color = PluginSettings.p60_fg_color
        self.p60_ui_refresh_hz = float(PluginSettings.p60_ui_refresh_hz)
        self.p60_use_opengl = bool(PluginSettings.p60_use_opengl)

        self.p66_show_legend = Var(bool(PluginSettings.p66_show_legend))
        self.p66_show_past_locations = Var(bool(PluginSettings.p66_show_past_locations))
        self.p66_show_global_plan = Var(bool(PluginSettings.p66_show_global_plan))
        self.p66_show_global_plan_boundaries = Var(bool(PluginSettings.p66_show_global_plan_boundaries))
        self.p66_show_local_plan = Var(bool(PluginSettings.p66_show_local_plan))
        self.p66_show_local_lattice = Var(bool(PluginSettings.p66_show_local_lattice))
        self.p66_show_state = Var(bool(PluginSettings.p66_show_state))
        self.p66_global_view_follow_planner = Var(bool(PluginSettings.p66_global_view_follow_planner))
        self.p66_frenet_view_follow_planner = Var(bool(PluginSettings.p66_frenet_view_follow_planner))
        self.p66_show_lidar_global = Var(bool(PluginSettings.p66_show_lidar_global))
        self.p66_show_lidar_frenet = Var(bool(PluginSettings.p66_show_lidar_frenet))
        self.p66_show_lidar_clusters = Var(bool(PluginSettings.p66_show_lidar_clusters))
        self.p66_show_race_boundary = Var(bool(PluginSettings.p66_show_race_boundary))
        self.p66_xy_zoom = float(PluginSettings.p66_xy_zoom)
        self.p66_frenet_zoom = float(PluginSettings.p66_frenet_zoom)
        self.p66_global_zoom = float(PluginSettings.p66_global_zoom)

        self.p67_show_local_global_view = Var(bool(PluginSettings.p67_show_local_global_view))
        self.p67_show_local_frenet_view = Var(bool(PluginSettings.p67_show_local_frenet_view))
        self.p67_show_occupancy_map = Var(bool(PluginSettings.p67_show_occupancy_map))
        self.p67_show_prediction = Var(bool(PluginSettings.p67_show_prediction))
        self.p67_global_plan_view = Var(bool(PluginSettings.p67_global_plan_view))
        self.p67_local_plan_view = Var(bool(PluginSettings.p67_local_plan_view))

        self.p68_show_core_logs = Var(bool(PluginSettings.p68_show_core_logs))
        self.p68_show_perceive_logs = Var(bool(PluginSettings.p68_show_perceive_logs))
        self.p68_show_plan_logs = Var(bool(PluginSettings.p68_show_plan_logs))
        self.p68_show_control_logs = Var(bool(PluginSettings.p68_show_control_logs))
        self.p68_show_execute_logs = Var(bool(PluginSettings.p68_show_execute_logs))
        self.p68_show_plugins_logs = Var(bool(PluginSettings.p68_show_plugins_logs))
        self.p68_disable_log = Var(bool(PluginSettings.p68_disable_log))
        self.p68_max_log_lines = int(PluginSettings.p68_max_log_lines)
        self.p68_log_pull_time = int(PluginSettings.p68_log_pull_time)

        self.vehicle_state = Var("Ego: —")
        self.lap = Var("0")
        self.current_wp = Var("0")
        self.elapsed_real_time = Var("0.00")
        self.elapsed_sim_time = Var("0.00")
        self.replan_fps = Var("0.0")
        self.control_fps = Var("0.0")
        self.perception_fps = Var("0.0")
        self.ui_fps = Var("0.0")
        self.plot_latency_ms = Var("0.0")

        self.perception_type = Var(ExecutionSettings.c40_perception or "")
        self.detection_strategy_type = Var(PerceptionSettings.c12_detection_strategy or "")
        self.tracking_strategy_type = Var(PerceptionSettings.c12_tracking_strategy or "")
        self.prediction_strategy_type = Var(PerceptionSettings.c12_prediction_strategy or "")
        self.localization_type = Var(ExecutionSettings.c40_localization or "")
        self.mapping_type = Var(ExecutionSettings.c40_mapping or "")
        self.global_planner_type = Var(ExecutionSettings.c40_global_planner or "")
        self.local_planner_type = Var(ExecutionSettings.c40_local_planner or "")
        self.behavioral_strategy_type = Var(PlanningSettings.c23_behavioral_strategy or "")
        self.path_strategy_type = Var(PlanningSettings.c23_path_strategy or "")
        self.velocity_strategy_type = Var(PlanningSettings.c23_velocity_strategy or "")
        self.controller_type = Var(ExecutionSettings.c40_controller or "")
        self.executer_type = Var(
            _strategy_default(ExecutionStrategy.registry, ExecutionSettings.c40_executer_type)
        )
        self.execution_bridge = Var(
            _strategy_default(WorldBridge.registry, ExecutionSettings.c40_bridge)
        )
        self.execution_tasks = Var(",".join(ExecutionSettings.c40_execution_tasks or []))
        self.probabilistic_collision_checking = Var(
            bool(PerceptionSettings.c15_probabilistic_collision_checking)
        )

        self.exec_plan = Var(True)
        self.exec_control = Var(True)
        self.exec_perceive = Var(True)
        self.exec_tasks = Var(True)
        self.exec_running = False

        self.control_dt = Var(float(ExecutionSettings.c40_control_dt))
        self.replan_dt = Var(float(ExecutionSettings.c40_replan_dt))
        self.perception_dt = Var(float(ExecutionSettings.c40_perception_dt))
        self.sim_dt = Var(float(ExecutionSettings.c40_sim_dt))
        self.localization_dt = Var(float(ExecutionSettings.c40_localization_dt))
        self.pace_perception = Var(bool(ExecutionSettings.c40_pace_perception))
        self.pace_replan = Var(bool(ExecutionSettings.c40_pace_replan))
        self.pace_control = Var(bool(ExecutionSettings.c40_pace_control))
        self.pace_sim = Var(bool(ExecutionSettings.c40_pace_sim))

        self.default_map_file = Var(ExecutionSettings.c40_map or "")
        self.default_global_plan_file = Var(ExecutionSettings.c40_global_trajectory or "")
        self.profile_list: list[str] = []
        self.cte_vel = Var(0.0)
        self.cte_steer = Var(0.0)
        self.cmd_accel = Var(0.0)
        self.cmd_steer = Var(0.0)

        self._wire_stack_writes()

    def _wire_stack_writes(self) -> None:
        def link(var: Var, assign: Callable[[], None]) -> None:
            def _cb() -> None:
                if self._syncing_stack:
                    return
                assign()

            var.bind(_cb)

        link(self.perception_type, lambda: setattr(ExecutionSettings, "c40_perception", self.perception_type.get()))
        link(self.localization_type, lambda: setattr(ExecutionSettings, "c40_localization", self.localization_type.get()))
        link(self.mapping_type, lambda: setattr(ExecutionSettings, "c40_mapping", self.mapping_type.get()))
        link(self.global_planner_type, lambda: setattr(ExecutionSettings, "c40_global_planner", self.global_planner_type.get()))
        link(self.local_planner_type, lambda: setattr(ExecutionSettings, "c40_local_planner", self.local_planner_type.get()))
        link(self.controller_type, lambda: setattr(ExecutionSettings, "c40_controller", self.controller_type.get()))
        link(self.executer_type, lambda: setattr(ExecutionSettings, "c40_executer_type", self.executer_type.get()))
        link(self.execution_bridge, lambda: setattr(ExecutionSettings, "c40_bridge", self.execution_bridge.get()))

        def _tasks() -> None:
            ExecutionSettings.c40_execution_tasks = list(self.execution_task_names())

        link(self.execution_tasks, _tasks)

        def _det() -> None:
            if self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_detection_strategy = self.detection_strategy_type.get()

        def _trk() -> None:
            if self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_tracking_strategy = self.tracking_strategy_type.get()

        def _prd() -> None:
            if self._syncing_perception_pipeline:
                return
            PerceptionSettings.c12_prediction_strategy = self.prediction_strategy_type.get()

        self.detection_strategy_type.bind(_det)
        self.tracking_strategy_type.bind(_trk)
        self.prediction_strategy_type.bind(_prd)

        def _beh() -> None:
            if self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_behavioral_strategy = self.behavioral_strategy_type.get()

        def _path() -> None:
            if self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_path_strategy = self.path_strategy_type.get()

        def _vel() -> None:
            if self._syncing_local_planning_pipeline:
                return
            PlanningSettings.c23_velocity_strategy = self.velocity_strategy_type.get()

        self.behavioral_strategy_type.bind(_beh)
        self.path_strategy_type.bind(_path)
        self.velocity_strategy_type.bind(_vel)

        link(
            self.probabilistic_collision_checking,
            lambda: setattr(
                PerceptionSettings,
                "c15_probabilistic_collision_checking",
                bool(self.probabilistic_collision_checking.get()),
            ),
        )

        for name, var in (
            ("c40_control_dt", self.control_dt),
            ("c40_replan_dt", self.replan_dt),
            ("c40_perception_dt", self.perception_dt),
            ("c40_sim_dt", self.sim_dt),
            ("c40_localization_dt", self.localization_dt),
        ):
            link(var, lambda n=name, v=var: setattr(ExecutionSettings, n, float(v.get())))

        for name, var in (
            ("c40_pace_perception", self.pace_perception),
            ("c40_pace_replan", self.pace_replan),
            ("c40_pace_control", self.pace_control),
            ("c40_pace_sim", self.pace_sim),
        ):
            link(var, lambda n=name, v=var: setattr(ExecutionSettings, n, bool(v.get())))

    def execution_task_names(self) -> list[str]:
        raw = str(self.execution_tasks.get() or "")
        return [p.strip() for p in raw.split(",") if p.strip()]

    def local_plan_view_enabled(self) -> bool:
        return bool(self.p67_show_local_global_view.get() or self.p67_show_local_frenet_view.get())

    def sync_app_from_singleton(self) -> None:
        self.c62_load_plugins.set(bool(AppSettings.c62_load_plugins))
        self.c60_selected_profile.set(AppSettings.c60_selected_profile or "default")

    def sync_stack_from_singletons(self) -> None:
        self._syncing_stack = True
        try:
            self.perception_type.set(ExecutionSettings.c40_perception or "")
            self.localization_type.set(ExecutionSettings.c40_localization or "")
            self.mapping_type.set(ExecutionSettings.c40_mapping or "")
            self.global_planner_type.set(ExecutionSettings.c40_global_planner or "")
            self.local_planner_type.set(ExecutionSettings.c40_local_planner or "")
            self.controller_type.set(ExecutionSettings.c40_controller or "")
            self.executer_type.set(
                _strategy_default(ExecutionStrategy.registry, ExecutionSettings.c40_executer_type)
            )
            self.execution_bridge.set(
                _strategy_default(WorldBridge.registry, ExecutionSettings.c40_bridge)
            )
            self.execution_tasks.set(",".join(ExecutionSettings.c40_execution_tasks or []))
            self.control_dt.set(float(ExecutionSettings.c40_control_dt))
            self.replan_dt.set(float(ExecutionSettings.c40_replan_dt))
            self.perception_dt.set(float(ExecutionSettings.c40_perception_dt))
            self.sim_dt.set(float(ExecutionSettings.c40_sim_dt))
            self.localization_dt.set(float(ExecutionSettings.c40_localization_dt))
            self.pace_perception.set(bool(ExecutionSettings.c40_pace_perception))
            self.pace_replan.set(bool(ExecutionSettings.c40_pace_replan))
            self.pace_control.set(bool(ExecutionSettings.c40_pace_control))
            self.pace_sim.set(bool(ExecutionSettings.c40_pace_sim))
            self.default_map_file.set(ExecutionSettings.c40_map or "")
            self.default_global_plan_file.set(ExecutionSettings.c40_global_trajectory or "")
            self.probabilistic_collision_checking.set(
                bool(PerceptionSettings.c15_probabilistic_collision_checking)
            )
            self._syncing_perception_pipeline = True
            try:
                self.detection_strategy_type.set(PerceptionSettings.c12_detection_strategy or "")
                self.tracking_strategy_type.set(PerceptionSettings.c12_tracking_strategy or "")
                self.prediction_strategy_type.set(PerceptionSettings.c12_prediction_strategy or "")
            finally:
                self._syncing_perception_pipeline = False
            self._syncing_local_planning_pipeline = True
            try:
                self.behavioral_strategy_type.set(PlanningSettings.c23_behavioral_strategy or "")
                self.path_strategy_type.set(PlanningSettings.c23_path_strategy or "")
                self.velocity_strategy_type.set(PlanningSettings.c23_velocity_strategy or "")
            finally:
                self._syncing_local_planning_pipeline = False
        finally:
            self._syncing_stack = False

    def registry_choices(self) -> dict[str, list[str]]:
        return {
            "perception": [""] + list(PerceptionStrategy.registry.keys()),
            "detection": list(DetectionStrategy.registry.keys()),
            "tracking": list(TrackingStrategy.registry.keys()),
            "prediction": [""] + list(PredictionStrategy.registry.keys()),
            "localization": [""] + list(LocalizationStrategy.registry.keys()),
            "mapping": [""] + list(MappingStrategy.registry.keys()),
            "global_planner": [""] + list(GlobalPlannerStrategy.registry.keys()),
            "local_planner": [""] + list(LocalPlanningStrategy.registry.keys()),
            "controller": [""] + list(ControlStrategy.registry.keys()),
            "executer": list(ExecutionStrategy.registry.keys()),
            "bridge": list(WorldBridge.registry.keys()),
        }

    # Compatibility for StackSettingsSync-style helpers used by the Tk binder.
    def push_to_stack(self) -> None:
        StackSettingsSync  # noqa: B018 — keep import used for type presence
