import race_plan_control.plan.trajectory as u
from race_plan_control.perceive.vehicle_state import VehicleState
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
        
        self.left_x, self.left_y = self.global_trajectory.convert_sd_path_to_xy_path(self.global_trajectory.path_s, self.ref_left_boundary_d)
        self.right_x, self.right_y = self.global_trajectory.convert_sd_path_to_xy_path(self.global_trajectory.path_s, self.ref_right_boundary_d)

        self.lap = 0


        self.xdata, self.ydata = [self.global_trajectory.path_x[0]], [self.global_trajectory.path_y[0]]
        self.past_d, self.past_s  = [self.global_trajectory.path_s[0]], [self.global_trajectory.path_d[0]]


        self.selected_edge:LatticeGraph.Edge = None
        self.lattice_graph = {} 
        self.lattice = LatticeGraph(self.global_trajectory, num_of_points=10)
        

    def reset(self,wp=0):
        self.xdata, self.ydata = [self.global_trajectory.path_x[wp]], [self.global_trajectory.path_y[wp]]
        self.past_s, self.past_d  = [self.global_trajectory.path_s[wp]], [self.global_trajectory.path_d[wp]]
        self.global_trajectory.update_waypoint_by_wp(wp)
        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
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
        if  self.selected_edge is not None and not self.selected_edge.local_trajectory.is_traversed(): 
            self.selected_edge.local_trajectory.update_to_next_waypoint()
            x_current, y_current = self.selected_edge.local_trajectory.get_current_xy()

        # nest edge selected, but finished
        elif self.selected_edge is not None and self.selected_edge.local_trajectory.is_traversed() and self.selected_edge.is_next_edge_selected():
            log.info("Edge Done, choosing next selected edge")
            self.selected_edge = self.selected_edge.selected_next_edge
            self.selected_edge.local_trajectory.update_to_next_waypoint()
            x_current, y_current = self.selected_edge.local_trajectory.get_current_xy()

        elif self.selected_edge is not None and self.selected_edge.local_trajectory.is_traversed() and not self.selected_edge.is_next_edge_selected():
            log.info("No next edge selected")
            x_current = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.path_y[self.global_trajectory.next_wp]
            self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")
            x_current = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.path_y[self.global_trajectory.next_wp]

        self.xdata.append(x_current)
        self.ydata.append(y_current)
        # TODO some error check might be needed
        self.global_trajectory.update_waypoint_by_xy(x_current, y_current)
        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_by_xy(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory.convert_xy_to_sd(x_current,y_current)
        self.past_d.append(d_)
        self.past_s.append(s_)
    
        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")
        
        return x_current, y_current

    def step(self, state:VehicleState):
        self.xdata.append(state.x)
        self.ydata.append(state.y)
        self.global_trajectory.update_waypoint_by_xy(state.x, state.y)


        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint_by_xy(state.x, state.y)

            if self.selected_edge.local_trajectory.is_traversed() and self.selected_edge.is_next_edge_selected():
                log.info("Edge Done, choosing next selected edge")
                self.selected_edge = self.selected_edge.selected_next_edge
                self.selected_edge.local_trajectory.update_to_next_waypoint()
            
            elif self.selected_edge.local_trajectory.is_traversed() and not self.selected_edge.is_next_edge_selected():
                self.selected_edge = None
        else:
            log.warning("No edge selected, back to closest next reference point")


        if self.global_trajectory.is_traversed():
            self.lap += 1
            log.info(f"Lap {self.lap} Done")
        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory.convert_xy_to_sd(state.x, state.y)
        self.past_d.append(d_)
        self.past_s.append(s_)

    
class LatticeGraph:
    def __init__(self, global_tj:u.Trajectory, num_of_points = 10, start_vel = None, end_vel = None):
        self.__global_tj = global_tj
        self.__num_of_points = num_of_points
        
        self.selected_edge:LatticeGraph.Edge = None
        self.lattice_graph = {} 

        self.__iter_idx = 0

    def add_edge(self, start_s, start_d, end_s, end_d):
        edge = self.Edge(start_s, start_d, end_s, end_d, num_of_points=self.__num_of_points)
        self.lattice_graph[edge] = self.__global_tj.create_trajectory_in_sd_coordinate(start_s, start_d, end_s, end_d, num_points=self.__num_of_points)

    def clear(self):
        self.lattice_graph = {}
        self.selected_edge = None
        self.__iter_idx = 0

    def __iter__(self):
        self.__iter_idx = 0
        return self

    def __next__(self):
        if self.__iter_idx < len(self.lattice_graph):
            self.__iter_idx += 1
            return self.lattice_graph[self.__iter_idx-1]
        else:
            raise StopIteration

    class Edge:
        def __init__(self, start_s, start_d, end_s, end_d, global_tj:u.Trajectory, num_of_points = 10):
            self.start_s = start_s
            self.start_d = start_d
            self.end_s = end_s
            self.end_d = end_d
            self.num_of_points = num_of_points
            self.local_trajectory = global_tj.create_trajectory_in_sd_coordinate(start_s, start_d, end_s, end_d, num_points=num_of_points)


            self.selected_next_edge = None
            self.next_edges = []
            self.is_selected = False
    
        def is_next_edge_selected(self):
            return self.selected_next_edge is not None

        def append_next_edges(self, edge):
            self.next_edges.append(edge)



        def __hash__(self):
            return hash((self.start_s, self.start_d, self.end_s, self.end_d, self.num_of_points))

        def __eq__(self, other):
            if isinstance(other, LatticeGraph.Edge):
                return (self.start_s == other.start_s and self.start_d == other.start_d and
                        self.end_s == other.end_s and self.end_d == other.end_d and
                        self.num_of_points == other.num_of_points)
            return False
        

    
if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()
