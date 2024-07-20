import race_plan_control.plan.trajectory as u
from race_plan_control.execute.vehicle_state import VehicleState
import numpy as np
import logging
log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import numpy as np
import race_plan_control.plan.trajectory as u


class Planner:
    def __init__(self, reference_path, ref_left_boundary_d, ref_right_boundary_d):
        self.reference_path = np.array(reference_path)
        self.global_trajectory = u.Trajectory(self.reference_path)
        

        self.ref_left_boundary_d = ref_left_boundary_d
        self.ref_right_boundary_d = ref_right_boundary_d
        
        self.left_x, self.left_y = self.global_trajectory._getXY_path(self.global_trajectory.reference_s, self.ref_left_boundary_d)
        self.right_x, self.right_y = self.global_trajectory._getXY_path(self.global_trajectory.reference_s, self.ref_right_boundary_d)

        self.lap = 0


        self.xdata, self.ydata = [self.global_trajectory.reference_x[0]], [self.global_trajectory.reference_y[0]]
        self.past_d, self.past_s  = [self.global_trajectory.reference_s[0]], [self.global_trajectory.reference_d[0]]

        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key

        self.selected_edge:Planner.EdgeManeuver = None
        
        

    def reset(self,wp=0):
        self.xdata, self.ydata = [self.global_trajectory.reference_x[wp]], [self.global_trajectory.reference_y[wp]]
        self.past_s, self.past_d  = [self.global_trajectory.reference_s[wp]], [self.global_trajectory.reference_d[wp]]
        self.global_trajectory.reset(wp)
        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
        self.selected_edge = None

    @abstractmethod
    def replan(self):
        pass


    def get_local_plan(self):
        if self.selected_edge is not None:
            log.info(f"Selected Edge: ({self.selected_edge.start_s:.2f},{self.selected_edge.start_d:.2f}) -> ({self.selected_edge.end_s:.2f},{self.selected_edge.end_d:.2f})")
            return self.selected_edge.local_trajectory
        return self.global_trajectory

    
    def step_wp(self):
        log.info(f"Step: {self.global_trajectory.current_wp}")
        if  self.selected_edge is not None and not self.selected_edge.local_trajectory.is_traversed(): 
            self.selected_edge.move_next_wp()
            x_current, y_current = self.selected_edge.get_current_xy()

        # nest edge selected, but finished
        elif self.selected_edge is not None and self.selected_edge.local_trajectory.is_traversed() and self.selected_edge.is_next_edge_selected():
            log.info("Edge Done, choosing next selected edge")
            self.selected_edge = self.selected_edge.selected_next_edge
            self.selected_edge.move_next_wp()
            x_current, y_current = self.selected_edge.get_current_xy()

        elif self.selected_edge is not None and self.selected_edge.local_trajectory.is_traversed() and not self.selected_edge.is_next_edge_selected():
            log.info("No next edge selected")
            x_current = self.global_trajectory.reference_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.reference_y[self.global_trajectory.next_wp]
            self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")
            x_current = self.global_trajectory.reference_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.reference_y[self.global_trajectory.next_wp]

        self.xdata.append(x_current)
        self.ydata.append(y_current)
        # TODO some error check might be needed
        self.global_trajectory.update_waypoint_xy(x_current, y_current)
        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_xy(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory._convert_to_frenet([(x_current,y_current)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])
    
        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

    def step(self, state:VehicleState):
        self.xdata.append(state.x)
        self.ydata.append(state.y)
        self.global_trajectory.update_waypoint_xy(state.x, state.y)

        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_xy(state.x, state.y)

            if self.selected_edge.local_trajectory.is_traversed() and self.selected_edge.is_next_edge_selected():
                log.info("Edge Done, choosing next selected edge")
                self.selected_edge = self.selected_edge.selected_next_edge
                self.selected_edge.move_next_wp()
            elif self.selected_edge.local_trajectory.is_traversed() and not self.selected_edge.is_next_edge_selected():
                self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")


        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")
        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory._convert_to_frenet([(state.x, state.y)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])

    
    class EdgeManeuver:
        def __init__(self, start_sd, end_sd, num_of_points = 10, start_vel = None, end_vel = None):
            self.start_s = start_sd[0] 
            self.start_d = start_sd[1]
            self.end_s = end_sd[0]
            self.end_d = end_sd[1]
            self.ts, self.td, self.tx, self.ty = None, None, None, None
            self.local_trajectory:u.Trajectory = None # same as a bove but

            self.current_idx = 0
            self.selected_next_edge = None
            self.next_edges = []

            self.is_selected = False
            
            self._num_of_points = num_of_points

        # tj is the race trajectory
        def generate_edge_trajectory(self,global_tj):
            # TODO to be optimized
            self.ts,self.td,self.tx,self.ty= global_tj.generate_local_edge_trajectory(self.start_s,self.end_s, self.start_d, self.end_d, num_points=self._num_of_points)
            self.local_trajectory = u.Trajectory(list(zip(self.tx,self.ty)), name="Local Trajectory")
            # If tj is null then we should generate wit respect to global coordinate
            
        def is_next_edge_selected(self):
            return self.selected_next_edge is not None

        def append_next_edges(self, edge):
            self.next_edges.append(edge)

        def get_current_sd(self):
            return self.ts[self.local_trajectory.current_wp], self.td[self.local_trajectory.current_wp]
        def get_current_xy(self):
            return self.tx[self.local_trajectory.current_wp], self.ty[self.local_trajectory.current_wp]
        
        def move_next_wp(self):
            self.local_trajectory.update_waypoint_wp(self.local_trajectory.current_wp + 1)

    
if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()