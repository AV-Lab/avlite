import race_plan_control.plan.trajectory as u
from race_plan_control.execute.vehicle_state import VehicleState
import numpy as np
import logging
log = logging.getLogger(__name__)


class Planner:
    def __init__(self, reference_path, ref_left_boundary_d, ref_right_boundary_d,
                 state:VehicleState = None ,  planning_horizon = 15, minimum_s_distance=5, minimum_boundary_distance=2):
        self.reference_path = np.array(reference_path)
        self.global_trajectory = u.Trajectory(self.reference_path)
        
        self.planning_horizon = planning_horizon
        self.minimum_planning_distance = minimum_s_distance
        self.minimum_boundary_distance = minimum_boundary_distance

        self.ref_left_boundary_d = ref_left_boundary_d
        self.ref_right_boundary_d = ref_right_boundary_d
        
        self.left_x, self.left_y = self.global_trajectory.getXY_path(self.global_trajectory.reference_s, self.ref_left_boundary_d)
        self.right_x, self.right_y = self.global_trajectory.getXY_path(self.global_trajectory.reference_s, self.ref_right_boundary_d)

        self.lap = 0


        if state is None:
            self.xdata, self.ydata = [self.global_trajectory.reference_x[0]], [self.global_trajectory.reference_y[0]]
            self.past_d, self.past_s  = [self.global_trajectory.reference_s[0]], [self.global_trajectory.reference_d[0]]
        else:
            self.xdata, self.ydata = [state.x], [state.y]
            s_, d_ = self.global_trajectory.convert_to_frenet([(state.x, state.y)])
            self.past_d, self.past_s  = [d_[0]], [s_[0]]

        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key

        self.selected_edge:Planner.EdgeManeuver = None
        
        

    def reset(self,wp=0):
        self.xdata, self.ydata = [self.global_trajectory.reference_x[wp]], [self.global_trajectory.reference_y[wp]]
        self.past_s, self.past_d  = [self.global_trajectory.reference_s[wp]], [self.global_trajectory.reference_d[wp]]
        self.global_trajectory.reset(wp)
        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
        self.selected_edge = None

    def replan(self, sample_size=2, back_to_ref_horizon=10, sample=True):
        if len(self.past_s) == 0:
            print("No data to replan")
            return

        s = self.past_s[-1]
        d = self.past_d[-1]
        
        # delete old edges that already passed its starting point
        self.lattice_graph = {k: v for k, v in self.lattice_graph.items() if s < v.start_s }

        ### Group 1
        # add edge on the reference trajectory
        idx = (self.global_trajectory.next_wp + back_to_ref_horizon)%len(self.global_trajectory.reference_s)
        next_s = self.global_trajectory.reference_s[idx]
        next_d = 0
        ep = Planner.EdgeManeuver((s,d), (next_s,next_d),num_of_points = back_to_ref_horizon+2) # +2 to include the start and end points
        ep.generate_edge_trajectory(self.global_trajectory)
        self.lattice_graph[(next_s,next_d)] = ep

        # sample new edges
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                s_ = self.past_s[-1] + s_e
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                ep = Planner.EdgeManeuver((s,d), (s_,d_))
                ep.generate_edge_trajectory(self.global_trajectory)
                self.lattice_graph[(s_,d_)] = ep
        ### Group 2
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                
                current_edges = list(self.lattice_graph.values())
                for e in current_edges:
                    s_ = e.end_s + s_e
                    s = e.end_s
                    d = e.end_d
                    ep = Planner.EdgeManeuver((s,d), (s_,d_))
                    ep.generate_edge_trajectory(self.global_trajectory)
                    e.append_next_edges(ep)
                    e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        
        ### Plan
        # Select a random edge from the lattice graph
        self.selected_edge = np.random.choice(list(self.lattice_graph.values()))
        self.selected_edge.selected_next_edge =  np.random.choice(self.selected_edge.next_edges) if len(self.selected_edge.next_edges) > 0 else None
 


    def get_local_plan(self):
        if self.selected_edge is not None:
            log.info(f"Selected Edge: ({self.selected_edge.start_s},{self.selected_edge.start_d}) -> ({self.selected_edge.end_s},{self.selected_edge.end_d})")
            return self.selected_edge.local_trajectory
        return self.global_trajectory



    
    def step(self):
        if  self.selected_edge is not None and not self.selected_edge.is_edge_traversed(): 
            self.selected_edge.next_idx()
            x_current, y_current = self.selected_edge.get_current_xy()

        # nest edge selected, but finished
        elif self.selected_edge is not None and self.selected_edge.is_edge_traversed() and self.selected_edge.is_next_edge_selected():
            log.info("Edge Done, choosing next selected edge")
            self.selected_edge = self.selected_edge.selected_next_edge
            self.selected_edge.next_idx()
            x_current, y_current = self.selected_edge.get_current_xy()

        elif self.selected_edge is not None and self.selected_edge.is_edge_traversed() and not self.selected_edge.is_next_edge_selected():
            log.info("No next edge selected")
            x_current = self.global_trajectory.reference_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.reference_y[self.global_trajectory.next_wp]
        else:
            log.warning("No edge selected, back to closest next reference point")
            x_current = self.global_trajectory.reference_x[self.global_trajectory.next_wp]
            y_current = self.global_trajectory.reference_y[self.global_trajectory.next_wp]

        self.xdata.append(x_current)
        self.ydata.append(y_current)
        # TODO some error check might be needed
        self.global_trajectory.update_waypoint(x_current, y_current)
        if self.selected_edge is not None:
            self.selected_edge.local_trajectory.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory.convert_to_frenet([(x_current,y_current)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])
    
        if len(self.past_d)>0:
            d_mean = sum(self.past_d) / len(self.past_d) 
            self.mse = sum((di - d_mean)**2 for di in self.past_d) / len(self.past_d) 

    def update_state(self, state):
        # if  self.selected_edge is not None and not self.selected_edge.is_edge_done(): 
            # self.selected_edge.next_idx()
        self.xdata.append(state.x)
        self.ydata.append(state.y)
        # TODO some error check might be needed
        self.global_trajectory.update_waypoint(state.x, state.y)
        if self.selected_edge is not None:
            log.info(f"Selected Edge: ({self.selected_edge.start_s},{self.selected_edge.start_d}) -> ({self.selected_edge.end_s},{self.selected_edge.end_d})")
            self.selected_edge.local_trajectory.update_waypoint(state.x, state.y)

            if self.selected_edge.is_edge_traversed() and self.selected_edge.is_next_edge_selected():
                log.info("Edge Done, choosing next selected edge")
                self.selected_edge = self.selected_edge.selected_next_edge
                self.selected_edge.next_idx()


        
        #### Frenet Coordinates
        s_, d_= self.global_trajectory.convert_to_frenet([(state.x, state.y)])
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
            self.num_of_points = num_of_points

            self.current_idx = 0
            self.selected_next_edge = None
            self.next_edges = []

        # tj is the race trajectory
        def generate_edge_trajectory(self,global_tj):
            # TODO to be optimized
            self.ts,self.td,self.tx,self.ty= global_tj.generate_local_edge_trajectory(self.start_s,self.end_s, self.start_d, self.end_d, num_points=self.num_of_points)
            self.local_trajectory = u.Trajectory(list(zip(self.tx,self.ty)), name="Local Trajectory")
            # If tj is null then we should generate wit respect to global coordinate
            
        def is_next_edge_selected(self):
            return self.selected_next_edge is not None

        def append_next_edges(self, edge):
            self.next_edges.append(edge)

        def get_current_sd(self):
            return self.ts[self.current_idx], self.td[self.current_idx]
        def get_current_xy(self):
            return self.tx[self.current_idx], self.ty[self.current_idx]
        
        def next_idx(self):
            if self.current_idx <= len(self.ts) - 1:
                self.current_idx += 1
            else:
                raise Exception("End of edge")

            return self.current_idx

        def is_edge_traversed(self):
            return self.current_idx >= len(self.ts) - 1

    
if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()