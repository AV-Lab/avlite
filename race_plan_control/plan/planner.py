import race_plan_control.plan.trajectory as u
from race_plan_control.perceive.vehicle_state import VehicleState
from race_plan_control.plan.lattice import LatticeGraph, EdgeTmp

import logging

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from race_plan_control.plan.trajectory import Trajectory


class Planner(ABC):
    global_trajectory: Trajectory
    ref_left_boundary_d: list[float]
    ref_right_boundary_d: list[float]
    ref_left_boundary_x: list[float]
    ref_left_boundary_y: list[float]
    ref_right_boundary_x: list[float]
    ref_right_boundary_y: list[float]
    lap: int
    traversed_x: list[float]
    traversed_y: list[float]
    traversed_d: list[float]
    traversed_s: list[float]

    def __init__(self, reference_path, ref_left_boundary_d, ref_right_boundary_d):
        self.global_trajectory = Trajectory(reference_path)
        self.ref_left_boundary_d = ref_left_boundary_d

        self.ref_right_boundary_d = ref_right_boundary_d

        self.ref_left_boundary_x, self.ref_left_boundary_y = (
            self.global_trajectory.convert_sd_path_to_xy_path(
                self.global_trajectory.path_s, self.ref_left_boundary_d
            )
        )
        self.ref_right_boundary_x, self.ref_right_boundary_y = (
            self.global_trajectory.convert_sd_path_to_xy_path(
                self.global_trajectory.path_s, self.ref_right_boundary_d
            )
        )

        self.lap = 0

        # these are localization data
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[0]], [
            self.global_trajectory.path_y[0]
        ]
        self.traversed_d, self.traversed_s = [self.global_trajectory.path_s[0]], [
            self.global_trajectory.path_d[0]
        ]

        self.selected_edge: EdgeTmp = None
        self.lattice_graph = {}
        self.lattice = LatticeGraph(self.global_trajectory, num_of_edge_points=10)

    def reset(self, wp=0):
        self.traversed_x, self.traversed_y = [self.global_trajectory.path_x[wp]], [
            self.global_trajectory.path_y[wp]
        ]
        self.traversed_s, self.traversed_d = [self.global_trajectory.path_s[wp]], [
            self.global_trajectory.path_d[wp]
        ]
        self.global_trajectory.update_waypoint_by_wp(wp)
        self.lattice_graph = (
            {}
        )  # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
        self.selected_edge = None

    @abstractmethod
    def replan(self):
        pass

    def get_local_plan(self):
        if self.selected_edge is not None:
            # log.info(f"Selected Edge: ({self.selected_edge.start_s:.2f},{self.selected_edge.start_d:.2f}) -> ({self.selected_edge.end_s:.2f},{self.selected_edge.end_d:.2f})")
            return self.selected_edge.local_trajectory
        return self.global_trajectory

    def step_wp(self):
        log.info(f"Step: {self.global_trajectory.current_wp}")
        if (
            self.selected_edge is not None
            and not self.selected_edge.local_trajectory.is_traversed()
        ):
            self.selected_edge.local_trajectory.update_to_next_waypoint()
            x_current, y_current = self.selected_edge.local_trajectory.get_current_xy()

        # nest edge selected, but finished
        elif (
            self.selected_edge is not None
            and self.selected_edge.local_trajectory.is_traversed()
            and self.selected_edge.is_next_edge_selected()
        ):
            log.info("Edge Done, choosing next selected edge")
            self.selected_edge = self.selected_edge.selected_next_edge
            self.selected_edge.local_trajectory.update_to_next_waypoint()
            x_current, y_current = self.selected_edge.local_trajectory.get_current_xy()

        elif (
            self.selected_edge is not None
            and self.selected_edge.local_trajectory.is_traversed()
            and not self.selected_edge.is_next_edge_selected()
        ):
            log.info("No next edge selected")
            x_current = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.path_y[self.global_trajectory.next_wp]
            self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")
            x_current = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.path_y[self.global_trajectory.next_wp]

        self.traversed_x.append(x_current)
        self.traversed_y.append(y_current)
        # TODO some error check might be needed
        self.global_trajectory.update_waypoint_by_xy(x_current, y_current)
        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_by_xy(
                x_current, y_current
            )

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(x_current, y_current)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)

        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

        return x_current, y_current

    def step(self, state: VehicleState):
        self.traversed_x.append(state.x)
        self.traversed_y.append(state.y)
        self.global_trajectory.update_waypoint_by_xy(state.x, state.y)

        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_by_xy(state.x, state.y)

            if (
                self.selected_edge.local_trajectory.is_traversed()
                and self.selected_edge.is_next_edge_selected()
            ):
                log.info("Edge Done, choosing next selected edge")
                self.selected_edge = self.selected_edge.selected_next_edge
                self.selected_edge.local_trajectory.update_to_next_waypoint()

            elif (
                self.selected_edge.local_trajectory.is_traversed()
                and not self.selected_edge.is_next_edge_selected()
            ):
                self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")

        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(state.x, state.y)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)


if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
