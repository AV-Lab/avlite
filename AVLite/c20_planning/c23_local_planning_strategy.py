from c10_perception.c12_perception_strategy import PerceptionModel
from c10_perception.c11_perception_model import EgoState
from c20_planning.c27_lattice import Edge, Lattice
from typing import Optional
from c20_planning.c28_trajectory import Trajectory, convert_sd_path_to_xy_path
from c20_planning.c21_planning_model import GlobalPlan
from abc import ABC, abstractmethod

import logging
log = logging.getLogger(__name__)


class LocalPlannerStrategy(ABC):


    registry = {}
    def __init__( self, global_plan: GlobalPlan, pm: PerceptionModel, planning_horizon=3, num_of_edge_points=10):
        """Initialize the local planner with a global plan and perception model."""
        self.global_plan: GlobalPlan = global_plan
        self.pm: PerceptionModel = pm
        self.global_trajectory: Trajectory = global_plan.trajectory
        self.traversed_x: list[float]
        self.traversed_y: list[float]
        self.traversed_d: list[float]
        self.traversed_s: list[float]
        self.location_xy: tuple[float, float]
        self.location_sd: tuple[float, float]
        self.lattice: Lattice

        self.selected_local_plan: Optional[Edge]
        self.planning_horizon: int
        self.num_of_edge_points: int

        # these are localization data
        self.traversed_x, self.traversed_y = [global_plan.start_point[0]], [global_plan.start_point[1]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])
        
        self.selected_local_plan: Optional[Edge] = None
        
        self.planning_horizon: int = planning_horizon
        self.num_of_edge_points: int = num_of_edge_points
        self.lattice: Lattice = Lattice( self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)
        self.lap: int = 0 


    def set_global_plan(self, global_plan: GlobalPlan) -> None:
        """
        Set the global plan for the local planner and initialize the trajectory.
        """
        self.global_plan = global_plan
        self.global_trajectory = global_plan.trajectory
        self.traversed_x, self.traversed_y = [global_plan.start_point[0]], [global_plan.start_point[1]]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]
        self.location_xy = (self.traversed_x[0], self.traversed_y[0])
        self.location_sd = (self.traversed_s[0], self.traversed_d[0])

        self.lattice: Lattice = Lattice( self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)

        log.info("Global Plan set and Local Planner reset.")


    def reset(self, wp:int=0):
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

    def get_local_plan(self) -> Trajectory:
        return self.selected_local_plan.local_trajectory if self.selected_local_plan is not None else self.global_trajectory

    def step_wp(self):
        """
        Advances the planner to the next waypoint and updates the traversed path.
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


    def step(self, state: EgoState):
        """
        Advances the planner based on the given vehicle state and updates the traversed path.

        Args:
            state: The current state of the vehicle.
        """
        # log.debug(f"Step called with state:  {state}")
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

    def local_plan_len(self, tmp_plan=None):
        edge = self.selected_local_plan if tmp_plan is None else tmp_plan
        return 1 + self.__plan_len(edge=edge.selected_next_local_plan)

    def __plan_len(self, edge):
        if edge is None:
            return 0
        return 1 + self.__plan_len(edge=edge.selected_next_local_plan)

