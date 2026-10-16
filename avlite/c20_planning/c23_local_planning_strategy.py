from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c29_settings import PlanningSettings, PlanningSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldRequirement
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

import logging
log = logging.getLogger(__name__)


class LocalPlanningStrategy(ABC):
    """Minimal local-planning interface.

    Owns localization bookkeeping (traversed path + current location in xy/sd)
    and the strategy registry. Concrete planners implement :meth:`replan` and
    return their result through :meth:`get_local_plan` as a :class:`LocalPlan`.

    The tick entrypoint :meth:`replan` takes optional ``perception_model`` and
    ``sensors``, supplied by the executer.

    Algorithm-specific machinery (e.g. lattice/edge search) lives in
    subclasses such as ``LatticePlanningStrategy``.
    """

    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset({
        StackCapability.GLOBAL_PLAN,
        StackCapability.LOCALIZATION,
    })
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self, global_plan: GlobalPlan, pm: PerceptionModel,
                 setting: PlanningSettingsSchema = PlanningSettings):
        """Initialize the local planner with a global plan and perception model."""
        self.global_plan: GlobalPlan = global_plan
        self.pm: PerceptionModel = pm
        self.global_trajectory: TrajectoryTracker = global_plan.trajectory

        self.traversed_x: list[float]
        self.traversed_y: list[float]
        self.traversed_d: list[float]
        self.traversed_s: list[float]
        self.location_xy: tuple[float, float]
        self.location_sd: tuple[float, float]

        # these are localization data
        self.traversed_x, self.traversed_y = [global_plan.start_point[0]], [global_plan.start_point[1]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])

        self.lap: int = 0

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy: Optional[tuple[float, float]] = None) -> None:
        """Set the global plan for the local planner and reset localization.

        ``ego_xy``: if provided, initialise Frenet location from the actual ego
        position rather than ``global_plan.start_point`` (which is always on the
        centerline, d=0).
        """
        if global_plan.trajectory is None:
            log.error("Global plan trajectory is None. Cannot set global plan.")
            return
        if len(global_plan.trajectory.path_s) == 0:
            log.error("Global plan trajectory is empty. Cannot set global plan.")
            return

        self.global_plan = global_plan
        self.global_trajectory = global_plan.trajectory
        ref_xy = ego_xy if ego_xy is not None else global_plan.start_point
        self.traversed_x, self.traversed_y = [ref_xy[0]], [ref_xy[1]]
        s0, d0 = self.global_trajectory.convert_xy_to_sd(*ref_xy)
        self.traversed_s, self.traversed_d = [s0], [d0]
        self.location_xy = (ref_xy[0], ref_xy[1])
        self.location_sd = (s0, d0)
        log.info(f"Global plan set: ego Frenet s={s0:.2f} d={d0:.2f}. Ego xy={ref_xy}. Global plan start={global_plan.start_point}")

    def reset(self, wp: int = 0):
        n = len(self.global_trajectory.path_x)
        if n == 0:
            self.traversed_x, self.traversed_y = [0.0], [0.0]
            self.traversed_s, self.traversed_d = [0.0], [0.0]
            self.location_xy = (0.0, 0.0)
            self.location_sd = (0.0, 0.0)
            return
        # Clamp: negative indices would silently select the track end via numpy
        # wrapping (Plan UI "step back" from wp 0), and oversized wp IndexError.
        wp = max(0, min(int(wp), n - 1))
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[wp]], [self.global_trajectory.path_y[wp]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[wp]], [self.global_trajectory.path_d[wp]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])
        self.global_trajectory.update_waypoint_by_wp(wp)

    @abstractmethod
    def replan(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ):
        """Run one local replan step.

        Args:
            perception_model: Stack world-state snapshot. When provided, becomes
                the authoritative model for this step (also stored on ``self.pm``).
                When omitted, use constructor-held ``self.pm``.
            sensors: World sensor snapshot for this tick (``None`` if unused).
        """
        pass

    def get_local_plan(self) -> LocalPlan:
        """Return the current local plan. Defaults to following the global trajectory."""
        return LocalPlan.from_trajectory(self.global_trajectory)

    def _advance_local_plan(self, state: EgoState) -> None:
        """Hook called from :meth:`step` after the global waypoint is updated.

        Subclasses override this to advance their own plan representation
        (e.g. a committed edge chain). The base implementation is a no-op.
        """

    def step_wp(self):
        """Advance the planner one waypoint along the global trajectory."""
        log.info(f"Step: {self.global_trajectory.current_wp}")
        x_new = self.global_trajectory.path_x[self.global_trajectory.next_wp]
        y_new = self.global_trajectory.path_y[self.global_trajectory.next_wp]

        self.traversed_x.append(x_new)
        self.traversed_y.append(y_new)
        self.global_trajectory.update_waypoint_by_xy(x_new, y_new)

        s_, d_ = self.global_trajectory.convert_xy_to_sd(x_new, y_new)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)

        if self.global_trajectory.is_traversed() and self.global_plan.race_mode:
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

        self.location_xy = (self.traversed_x[-1], self.traversed_y[-1])
        self.location_sd = (self.traversed_s[-1], self.traversed_d[-1])

    def step(self, state: EgoState):
        """Advance the planner based on the given vehicle state.

        Updates the traversed path, the closest global waypoint, frenet
        coordinates, and lap counting. Algorithm-specific plan advancement is
        delegated to :meth:`_advance_local_plan`.
        """
        self.traversed_x.append(state.x)
        self.traversed_y.append(state.y)
        self.global_trajectory.update_waypoint_by_xy(state.x, state.y)

        self._advance_local_plan(state)

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(state.x, state.y)

        # Lap detection via S-coordinate crossover.
        # global_trajectory.is_traversed() relies on current_wp reaching the last index,
        # which is unreliable when the ego is laterally displaced and the closest global
        # waypoint jumps directly from near-end to near-start. Instead, compare the
        # previous S value to the new one: if we were near the end of the track (s > 80%)
        # and are now near the start (s < 5%), a lap has been completed.
        if self.global_plan.race_mode and len(self.traversed_s) > 0:
            track_len = self.global_trajectory.track_end_s
            if track_len > 0 and self.traversed_s[-1] > track_len * 0.8 and s_ < track_len * 0.05:
                self.lap += 1
                log.info(f"Lap {self.lap} Done")

        self.traversed_d.append(d_)
        self.traversed_s.append(s_)
        self.location_xy = (state.x, state.y)
        self.location_sd = (s_, d_)

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalPlanningStrategy.registry[cls.__name__] = cls


class LocalBehavioralPlanningStrategy(ABC):
    """Behavioral planning stage: decides high-level driving intent.

    Consumes and returns the shared :class:`LocalPlan` working object, setting
    :attr:`LocalPlan.behavior`. Mirrors the perception sub-strategy pattern.
    """

    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset()

    @abstractmethod
    def plan_behavior(self, plan: LocalPlan) -> LocalPlan:
        """Decide the driving intent and store it on ``plan.behavior``."""
        raise NotImplementedError

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalBehavioralPlanningStrategy.registry[cls.__name__] = cls


class LocalPathPlanningStrategy(ABC):
    """Path planning stage: produces the geometric path for the local plan."""

    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.LOCAL_PLAN})

    @abstractmethod
    def plan_path(self, plan: LocalPlan) -> LocalPlan:
        """Fill ``plan.path``/``plan.trajectory`` with the planned geometry."""
        raise NotImplementedError

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalPathPlanningStrategy.registry[cls.__name__] = cls


class LocalVelocityPlanningStrategy(ABC):
    """Velocity planning stage: produces the velocity profile for the local plan."""

    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.LOCAL_PLAN})

    @abstractmethod
    def plan_velocity(self, plan: LocalPlan) -> LocalPlan:
        """Fill ``plan.velocity`` with the planned speed profile."""
        raise NotImplementedError

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalVelocityPlanningStrategy.registry[cls.__name__] = cls


class LocalPlanningPipeline(LocalPlanningStrategy):
    """Pipelined local-planning strategy: behavioral -> path -> velocity.

    Each stage is resolved by name from its registry at construction time.
    An empty name means that stage is skipped. A single mutable
    :class:`LocalPlan` working object is threaded through the stages, mirroring
    how :class:`PerceptionPipeline` threads a ``PerceptionModel``.
    """

    def __init__(self, global_plan: GlobalPlan, env: PerceptionModel,
                 setting: PlanningSettingsSchema = PlanningSettings):
        super().__init__(global_plan=global_plan, pm=env, setting=setting)
        self._behavioral = self._resolve(
            LocalBehavioralPlanningStrategy.registry, setting.c23_behavioral_strategy,
            global_plan, env, setting)
        self._path = self._resolve(
            LocalPathPlanningStrategy.registry, setting.c23_path_strategy,
            global_plan, env, setting)
        self._velocity = self._resolve(
            LocalVelocityPlanningStrategy.registry, setting.c23_velocity_strategy,
            global_plan, env, setting)
        self._working_plan: Optional[LocalPlan] = None

    @staticmethod
    def _resolve(registry: dict, name: str, global_plan: GlobalPlan, env: PerceptionModel,
                 setting: PlanningSettingsSchema):
        """Instantiate a stage by name, passing only the constructor args it accepts."""
        if not name or name not in registry:
            return None
        cls = registry[name]
        available = {
            "global_plan": global_plan,
            "env": env,
            "pm": env,
            "setting": setting,
        }
        try:
            params = inspect.signature(cls.__init__).parameters
        except (ValueError, TypeError):
            return cls()
        kwargs = {k: v for k, v in available.items() if k in params}
        return cls(**kwargs)

    @property
    def _stages(self):
        return (self._behavioral, self._path, self._velocity)

    @property
    def world_requirements(self) -> frozenset[WorldRequirement]:
        reqs: set[WorldRequirement] = set()
        for stage in self._stages:
            if stage is not None:
                reqs |= stage.world_requirements
        return frozenset(reqs)

    @property
    def stack_requirements(self) -> frozenset[StackRequirement]:
        reqs: set[StackRequirement] = {StackCapability.GLOBAL_PLAN, StackCapability.LOCALIZATION}
        for stage in self._stages:
            if stage is not None:
                reqs |= stage.stack_requirements
        return frozenset(reqs)

    @property
    def stack_capabilities(self) -> frozenset[StackCapability]:
        return frozenset({StackCapability.LOCAL_PLAN})

    def _child_planners(self):
        """Stages that are also LocalPlanningStrategy instances (own localization)."""
        return [s for s in self._stages if isinstance(s, LocalPlanningStrategy)]

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        for stage in self._child_planners():
            stage.set_global_plan(global_plan, ego_xy=ego_xy)
        self._working_plan = None

    def reset(self, wp: int = 0):
        super().reset(wp)
        for stage in self._child_planners():
            stage.reset(wp)
        self._working_plan = None

    def step(self, state: EgoState):
        super().step(state)
        for stage in self._child_planners():
            stage.step(state)

    def step_wp(self):
        super().step_wp()
        for stage in self._child_planners():
            stage.step_wp()

    def replan(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ):
        if perception_model is not None:
            self.pm = perception_model
        plan = LocalPlan.from_trajectory(self.global_trajectory)
        if self._behavioral is not None:
            plan = self._behavioral.plan_behavior(plan)
        if self._path is not None:
            plan = self._path.plan_path(plan)
        if self._velocity is not None:
            plan = self._velocity.plan_velocity(plan)
        self._working_plan = plan

    def get_local_plan(self) -> LocalPlan:
        if self._working_plan is not None:
            return self._working_plan
        return super().get_local_plan()
