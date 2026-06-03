from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Type

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c29_settings import PlanningSettings

if TYPE_CHECKING:
    from avlite.c30_control.c32_control_strategy import ControlStrategy

import logging
log = logging.getLogger(__name__)


class LocalPlanningStrategy(ABC):
    """Minimal local-planning interface.

    Owns localization bookkeeping (traversed path + current location in xy/sd)
    and the strategy registry. Concrete planners implement :meth:`replan` and
    return their result through :meth:`get_local_plan` as a :class:`LocalPlan`.

    Algorithm-specific machinery (e.g. lattice/edge search) lives in
    subclasses such as ``LatticePlanningStrategy``.
    """

    registry = {}

    def __init__(self, global_plan: GlobalPlan, pm: PerceptionModel,
                 controller: Optional['ControlStrategy'] = None,
                 setting: Type[PlanningSettings] = PlanningSettings):
        """Initialize the local planner with a global plan and perception model."""
        self.global_plan: GlobalPlan = global_plan
        self.pm: PerceptionModel = pm
        self.controller: Optional['ControlStrategy'] = controller
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

    def set_global_plan(self, global_plan: GlobalPlan) -> None:
        """Set the global plan for the local planner and reset localization."""
        if global_plan.trajectory is None:
            log.error("Global plan trajectory is None. Cannot set global plan.")
            return
        if len(self.global_plan.trajectory.path_s) == 0:
            log.error("Global plan trajectory is empty. Cannot set global plan.")
            return

        self.global_plan = global_plan
        self.global_trajectory = global_plan.trajectory
        self.traversed_x, self.traversed_y = [global_plan.start_point[0]], [global_plan.start_point[1]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])

        log.info("Global Plan set and Local Planner reset.")

    def reset(self, wp: int = 0):
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[wp]], [self.global_trajectory.path_y[wp]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[wp]], [self.global_trajectory.path_d[wp]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])
        self.global_trajectory.update_waypoint_by_wp(wp)

    @abstractmethod
    def replan(self):
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
            track_len = self.global_trajectory.path_s[-2]
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
