import json
import numpy as np

import plan.race_trajectory as u
import sys
import logging



class Planner:
    def __init__(self, path_to_track, planning_horizon = 15, minimum_s_distance=5, minimum_boundary_distance=2):
        with open(path_to_track, 'r') as f:
            track_data = json.load(f)
        logging.info(f"Track data loaded from {path_to_track}")

        self.planning_horizon = planning_horizon
        self.minimum_planning_distance = minimum_s_distance
        self.minimum_boundary_distance = minimum_boundary_distance


        self.reference_path = np.array([point[:2] for point in track_data["ReferenceLine"]])
        self.reference_x = [point[0] for point in track_data["ReferenceLine"]]
        self.reference_y = [point[1] for point in track_data["ReferenceLine"]]
        self.race_trajectory = u.trajectory(self.reference_path)
        self.reference_s, self.reference_d = self.race_trajectory.convert_to_frenet(self.reference_path)

        self.ref_left_boundary_d = track_data["LeftBound"]
        self.ref_right_boundary_d = track_data["RightBound"]
        self.left_x, self.left_y = self.race_trajectory.getXY_path(self.reference_s, self.ref_left_boundary_d)
        self.right_x, self.right_y = self.race_trajectory.getXY_path(self.reference_s, self.ref_right_boundary_d)

        self.x_vel = 0
        self.y_vel = 0

        self.xdata, self.ydata = [self.reference_x[0]], [self.reference_y[0]]
        self.past_d, self.past_s  = [self.reference_s[0]], [self.reference_d[0]]
        self.mse = 0
        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
        self.selected_edge = None

    def reset(self,wp=None):
        if wp is None:
            wp = 0
        self.xdata, self.ydata = [self.reference_x[wp]], [self.reference_y[wp]]
        self.past_d, self.past_s  = [self.reference_s[wp]], [self.reference_d[wp]]
        self.race_trajectory.reset(wp)
        self.mse = 0
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
        idx = (self.race_trajectory.next_wp + back_to_ref_horizon)%len(self.reference_s)
        next_s = self.reference_s[idx]
        next_d = 0
        ep = Planner.edge_maneuver((s,d), (next_s,next_d),num_of_points = back_to_ref_horizon+2) # +2 to include the start and end points
        ep.generate_edge_trajectory(self.race_trajectory)
        self.lattice_graph[(next_s,next_d)] = ep

        # sample new edges
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                s_ = self.past_s[-1] + s_e
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                ep = Planner.edge_maneuver((s,d), (s_,d_))
                ep.generate_edge_trajectory(self.race_trajectory)
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
                    ep = Planner.edge_maneuver((s,d), (s_,d_))
                    ep.generate_edge_trajectory(self.race_trajectory)
                    e.append_next_edges(ep)
                    e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        
        ### Plan
        # Select a random edge from the lattice graph
        self.selected_edge = np.random.choice(list(self.lattice_graph.values()))
        self.selected_edge.selected_next_edge =  np.random.choice(self.selected_edge.next_edges) if len(self.selected_edge.next_edges) > 0 else None
 


    def get_local_plan(self, horizon=10):
        t = self.race_trajectory.next_wp - 1
        return self.race_trajectory[t:t+horizon]



    def step_at_fixed_loc(self, x_current, y_current):
        self.xdata.append(x_current)
        self.ydata.append(y_current)
        if self.selected_edge is not None and not self.selected_edge.is_edge_done(): 
            self.selected_edge.next_idx()
        # TODO: Else need replan

        self.race_trajectory.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.race_trajectory.convert_to_frenet([(x_current,y_current)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])
        
        if len(self.past_d)>0:
            d_mean = sum(self.past_d) / len(self.past_d) 
            self.mse = sum((di - d_mean)**2 for di in self.past_d) / len(self.past_d) 
    # TODO FSM to be carefully thought out

    def step(self):
        if  self.selected_edge is not None and not self.selected_edge.is_edge_done(): 
            self.selected_edge.next_idx()
            x_current, y_current = self.selected_edge.get_current_xy()

        elif self.selected_edge is not None and self.selected_edge.is_edge_done() and self.selected_edge.is_next_edge_selected():
            print("Edge Done, choosing next selected edge")
            self.selected_edge = self.selected_edge.get_next_edge()
            x_current, y_current = self.selected_edge.get_current_xy()

        elif self.selected_edge is not None and self.selected_edge.is_edge_done() and not self.selected_edge.is_next_edge_selected():
            print("No next edge selected")
            x_current = self.reference_x[self.race_trajectory.next_wp]
            y_current = self.reference_y[self.race_trajectory.next_wp]
        else:
            print("No edge selected, back to closest next reference point")
            x_current = self.reference_x[self.race_trajectory.next_wp]
            y_current = self.reference_y[self.race_trajectory.next_wp]

        logging.info("step called")
        self.xdata.append(x_current)
        self.ydata.append(y_current)
        # TODO some error check might be needed
        self.race_trajectory.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.race_trajectory.convert_to_frenet([(x_current,y_current)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])
    
        if len(self.past_d)>0:
            d_mean = sum(self.past_d) / len(self.past_d) 
            self.mse = sum((di - d_mean)**2 for di in self.past_d) / len(self.past_d) 

        
    
    class edge_maneuver:
        def __init__(self, start_sd, end_sd, num_of_points = 10, start_vel = None, end_vel = None):
            self.start_s = start_sd[0] 
            self.start_d = start_sd[1]
            self.end_s = end_sd[0]
            self.end_d = end_sd[1]
            self.ts, self.td, self.tx, self.ty = None, None, None, None
            self.num_of_points = num_of_points

            self.current_idx = 0
            self.selected_next_edge = None
            self.next_edges = []

        # tj is the race trajectory
        def generate_edge_trajectory(self,tj):
            self.ts,self.td,self.tx,self.ty= tj.generate_local_edge_trajectory(self.start_s,self.end_s, self.start_d, self.end_d, num_points=self.num_of_points)
            # If tj is null then we should generate wit respect to global coordinate
            
        def get_next_edge(self):
            return self.selected_next_edge
        def set_next_edge(self, edge):
            self.selected_next_edge = edge
        
        def is_next_edge_selected(self):
            return self.selected_next_edge is not None

        def append_next_edges(self, edge):
            self.next_edges.append(edge)

        def get_current_sd(self):
            return self.ts[self.current_idx], self.td[self.current_idx]
        def get_current_xy(self):
            return self.tx[self.current_idx], self.ty[self.current_idx]
        
        def next_idx(self):
            if self.current_idx < len(self.ts) - 1:
                self.current_idx += 1
            else:
                raise Exception("End of edge")

            return self.current_idx

        def is_edge_done(self):
            return self.current_idx >= len(self.ts) - 1

    
import visualizer 
if __name__ == "__main__":
    sys.path.append("..")
    visualizer.main()