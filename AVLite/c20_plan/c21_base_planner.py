from c10_perceive.c11_perception_model import PerceptionModel
from c10_perceive.c12_state import EgoState
from c20_plan.c23_lattice import Edge, Lattice
from typing import Optional
from c20_plan.c24_trajectory import Trajectory
from abc import ABC, abstractmethod

import logging
log = logging.getLogger(__name__)


class BasePlanner(ABC):
    global_trajectory: Trajectory
    ref_left_boundary_d: list[float]
    ref_right_boundary_d: list[float]
    ref_left_boundary_x: list[float]
    ref_left_boundary_y: list[float]
    ref_right_boundary_x: list[float]
    ref_right_boundary_y: list[float]
    traversed_x: list[float]
    traversed_y: list[float]
    traversed_d: list[float]
    traversed_s: list[float]
    location_xy: tuple[float, float]
    location_sd: tuple[float, float]
    lap: int = 0 

    def __init__(
        self,
        global_path: list[tuple[float, float]],
        ref_left_boundary_d: list[float],
        ref_right_boundary_d: list[float],
        env: PerceptionModel,
        planning_horizon=3,
        num_of_edge_points=10,
    ):
        
        self._env = env


        self.global_trajectory = Trajectory(global_path)

        self.ref_left_boundary_d = ref_left_boundary_d
        self.ref_right_boundary_d = ref_right_boundary_d

        self.ref_left_boundary_x, self.ref_left_boundary_y = self.global_trajectory.convert_sd_path_to_xy_path(
            self.global_trajectory.path_s, self.ref_left_boundary_d
        )
        self.ref_right_boundary_x, self.ref_right_boundary_y = self.global_trajectory.convert_sd_path_to_xy_path(
            self.global_trajectory.path_s, self.ref_right_boundary_d
        )

        # these are localization data
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[0]], [self.global_trajectory.path_y[0]]
        self.traversed_d, self.traversed_s = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])
        
        self.selected_local_plan: Optional[Edge] = None
        
        self.planning_horizon: int = planning_horizon
        self.num_of_edge_points: int = num_of_edge_points
        self.lattice = Lattice(
            self.global_trajectory,
            self.ref_left_boundary_d,
            self.ref_right_boundary_d,
            planning_horizon=self.planning_horizon,
            num_of_points=self.num_of_edge_points,
        )


    def reset(self, wp=0):
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[wp]], [self.global_trajectory.path_y[wp]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[wp]], [self.global_trajectory.path_d[wp]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])
        self.global_trajectory.update_waypoint_by_wp(wp)
        self.selected_local_plan = None
        self.lattice.reset()

    @abstractmethod
    def replan(self):
        pass

    def get_local_plan(self):
        if self.selected_local_plan is not None:
            # log.info(f"Selected Edge: ({self.selected_edge.start_s:.2f},{self.selected_edge.start_d:.2f}) -> ({self.selected_edge.end_s:.2f},{self.selected_edge.end_d:.2f})")
            return self.selected_local_plan.local_trajectory
        return self.global_trajectory

    def step_wp(self):
        """
        Advances the planner to the next waypoint and updates the traversed path.

        Returns:
            The current x and y coordinates after stepping to the next waypoint.
        """
        log.info(f"Step: {self.global_trajectory.current_wp}")
        # next edge selected, but not finished
        if self.selected_local_plan is not None and not self.selected_local_plan.local_trajectory.is_traversed():
            self.selected_local_plan.local_trajectory.update_to_next_waypoint()
            x_new, y_new = self.selected_local_plan.local_trajectory.get_current_xy()

        # next edge selected, but finished
        elif (
            self.selected_local_plan is not None
            and self.selected_local_plan.local_trajectory.is_traversed()
            and self.selected_local_plan.selected_next_local_plan is not None
        ):
            log.info("Local Plan Completed, choosing next selected Local Plan")
            self.selected_local_plan = self.selected_local_plan.selected_next_local_plan
            self.selected_local_plan.local_trajectory.update_to_next_waypoint()
            x_new, y_new = self.selected_local_plan.local_trajectory.get_current_xy()
        # no edge selected
        elif (
            self.selected_local_plan is not None
            and self.selected_local_plan.local_trajectory.is_traversed()
            and self.selected_local_plan.selected_next_local_plan is None
        ):
            log.info("Local Plan Traversed. No next Local Plan selected")
            x_new = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_new = self.global_trajectory.path_y[self.global_trajectory.next_wp]
            self.selected_local_plan = None
        else:
            log.warning("No Local Plan, back to closest next reference point")
            x_new = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_new = self.global_trajectory.path_y[self.global_trajectory.next_wp]

        self.traversed_x.append(x_new)
        self.traversed_y.append(y_new)

        # TODO some error check might be needed
        self.global_trajectory.update_waypoint_by_xy(x_new, y_new)
        if self.selected_local_plan is not None:
            self.selected_local_plan.local_trajectory.update_waypoint_by_xy(x_new, y_new)

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(x_new, y_new)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)

        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")
        
        self.location_xy = (self.traversed_x[-1], self.traversed_y[-1])
        self.location_sd = (self.traversed_s[-1], self.traversed_d[-1])

        return x_new, y_new

    def step(self, state: EgoState):
        """
        Advances the planner based on the given vehicle state and updates the traversed path.

        Args:
            state: The current state of the vehicle.
        """
        self.traversed_x.append(state.x)
        self.traversed_y.append(state.y)
        self.global_trajectory.update_waypoint_by_xy(state.x, state.y)

        if self.selected_local_plan is not None:
            
            self.selected_local_plan.local_trajectory.update_waypoint_by_xy(state.x, state.y)

            if self.selected_local_plan.local_trajectory.is_traversed() and self.selected_local_plan.selected_next_local_plan is not None:
                log.info("Local Plan Traversed, choosing next selected Local Plan")
                self.selected_local_plan = self.selected_local_plan.selected_next_local_plan
                self.selected_local_plan.local_trajectory.update_to_next_waypoint()

            elif self.selected_local_plan.local_trajectory.is_traversed() and self.selected_local_plan.selected_next_local_plan is None:
                log.info("Local plan traversed, no next local plan selected. I'll follow the global trajectory")
                self.selected_local_plan = None

        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(state.x, state.y)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)
        self.location_xy = (state.x, state.y)
        self.location_sd = (s_, d_)
