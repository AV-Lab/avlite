import matplotlib.pyplot as plt
import json
import numpy as np

import matplotlib
import race_trajectory as u
import time

matplotlib.use("TkAgg")


class planner:
    def __init__(self, path_to_track, frenet_zoom=15, xy_zoom=30, planning_horizon = 15, minimum_s_distance=5, minimum_boundary_distance=2):
        self.planning_horizon = planning_horizon
        self.minimum_planning_distance = minimum_s_distance
        self.minimum_boundary_distance = minimum_boundary_distance
        self.frenet_zoom = frenet_zoom
        self.xy_zoom = xy_zoom

        with open(path_to_track, 'r') as f:
            track_data = json.load(f)

        self.reference_path = np.array([point[:2] for point in track_data["ReferenceLine"]])
        self.reference_x = [point[0] for point in track_data["ReferenceLine"]]
        self.reference_y = [point[1] for point in track_data["ReferenceLine"]]
        self.tj = u.trajectory(self.reference_path)
        self.reference_s, self.reference_d = self.tj.convert_to_frenet(self.reference_path)

        self.ref_left_boundary_d = track_data["LeftBound"]
        self.ref_right_boundary_d = track_data["RightBound"]
        self.left_x, self.left_y = self.tj.getXY_path(self.reference_s, self.ref_left_boundary_d)
        self.right_x, self.right_y = self.tj.getXY_path(self.reference_s, self.ref_right_boundary_d)

        self.x_vel = 0
        self.y_vel = 0

        self.xdata, self.ydata = [], []
        self.past_d = []
        self.past_s = []
        self.mse = 0
        self.prev_time = None

        self.lattice_graph = {} # intended to hold local plan lattice graph. A dictionary with source (s,d) as key
        self.selected_edge = None
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)

        
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
        idx = (self.tj.next_wp + back_to_ref_horizon)%len(self.reference_s)
        next_s = self.reference_s[idx]
        next_d = 0
        ep = edge_maneuver((s,d), (next_s,next_d),num_of_points = back_to_ref_horizon+2) # +2 to include the start and end points
        ep.generate_edge_trajectory(self.tj)
        self.lattice_graph[(next_s,next_d)] = ep

        # sample new edges
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                s_ = self.past_s[-1] + s_e
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                ep = edge_maneuver((s,d), (s_,d_))
                ep.generate_edge_trajectory(self.tj)
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
                    ep = edge_maneuver((s,d), (s_,d_))
                    ep.generate_edge_trajectory(self.tj)
                    e.append_next_edges(ep)
                    e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        
        ### Plan
        
        # Select a random edge from the lattice graph
        self.selected_edge = np.random.choice(list(self.lattice_graph.values()))
        self.selected_edge.selected_next_edge =  np.random.choice(self.selected_edge.next_edges) if len(self.selected_edge.next_edges) > 0 else None
 

    def step_at_fixed_loc(self, x_current, y_current):
        self.xdata.append(x_current)
        self.ydata.append(y_current)
        if self.selected_edge is not None and not self.selected_edge.is_edge_done(): 
            self.selected_edge.next_idx()
        # TODO: Else need replan

        self.tj.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.tj.convert_to_frenet([(x_current,y_current)])
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
            x_current = self.reference_x[self.tj.next_wp]
            y_current = self.reference_y[self.tj.next_wp]
        else:
            print("No edge selected, back to closest next reference point")
            x_current = self.reference_x[self.tj.next_wp]
            y_current = self.reference_y[self.tj.next_wp]


        self.xdata.append(x_current)
        self.ydata.append(y_current)
        # TODO some error check might be needed
        self.tj.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.tj.convert_to_frenet([(x_current,y_current)])
        self.past_d.append(d_[0])
        self.past_s.append(s_[0])
    
        if len(self.past_d)>0:
            d_mean = sum(self.past_d) / len(self.past_d) 
            self.mse = sum((di - d_mean)**2 for di in self.past_d) / len(self.past_d) 

        
    def plot(self, pause_interval = 0.01):
        if len(self.xdata) == 0:
            print("No data to plot")
            return
        self.ax1.clear()
        self.ax2.clear()
        

        # Plot track boundaries in black

        self.ax1.plot(self.left_x,self.left_y, color='orange', label='Left Boundary')  # Change color and label as needed
        self.ax1.plot(self.right_x, self.right_y, color='tan', label='Right Boundary')  # Change color and label as needed
        

        # Plot the reference path in blue
        self.ax1.plot(self.reference_x, self.reference_y, 'b', label="Reference Trajectory")  # reference path in blue
        # self.ax1.scatter(self.reference_x, self.reference_y, color="blue", label="Reference Trajectory")  # reference path in blue
        
        # For Zoom in
        if self.xy_zoom is not None:
            range = self.xy_zoom
            self.ax1.set_xlim(self.xdata[-1] - range, self.xdata[-1] + range)
            self.ax1.set_ylim(self.ydata[-1] - range, self.ydata[-1] + range)
        
        self.ax1.plot(self.xdata, self.ydata, 'r-', label='Past Locations')  # Plot all points in red
        self.ax1.plot(self.xdata[-100:], self.ydata[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
        self.ax1.plot(self.xdata[-1], self.ydata[-1], 'ro', markersize=10, label='Current Location')  

        if self.selected_edge is not None:
            x,y = self.selected_edge.get_current_xy() 
            s,d = self.selected_edge.get_current_sd()
            self.ax1.plot(x,y, 'bo', markersize=7, label='Planned Location')  
            self.ax2.plot(s,d, 'bo', markersize=9, label='Planned Location')  
            
        
        if self.tj.next_wp is not None: 
            self.ax1.plot(self.reference_x[self.tj.next_wp], self.reference_y[self.tj.next_wp], 'gx', markersize=10, label='Next WP')  
            self.ax2.plot(self.reference_s[self.tj.next_wp], self.reference_d[self.tj.next_wp], 'gx', markersize=10, label='Next WP')  


        # Use ax2 for the Frenet coordinates
        
        # self.ax2.axhline(0, color='blue', linewidth=0.5, label='Reference Path') 
        self.ax2.scatter(self.reference_s, self.reference_d, s=5, alpha=.5, color="blue", label="Reference Trajectory")  # reference path in blue
       

        current_time = time.time()
        if self.prev_time is not None:
            self.x_vel = (self.past_s[-1] - self.past_s[-2]) / (current_time - self.prev_time)
            self.y_vel = (self.past_d[-1] - self.past_d[-2]) / (current_time - self.prev_time)
            self.ax2.text(self.past_s[-1]-1.1, self.past_d[-1]+1, f'X Velocity: {self.x_vel:.2f}\nY Velocity: {self.y_vel:.2f}', verticalalignment='bottom', fontsize=9)
            self.ax2.text(self.past_s[-1]-1.1, self.past_d[-1]-1, f'Current d: {self.past_d[-1]:.2f}\nMSE: {self.mse:.2f}',
                        verticalalignment='top', fontsize=12)
        self.prev_time = current_time
       
        if not len(self.past_s)==0:
            self.ax2.plot(self.past_s, self.past_d, 'r-', label='Past Locations')  # Plot all points in red
            self.ax2.plot(self.past_s[-100:], self.past_d[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
            self.ax2.plot(self.past_s[-1], self.past_d[-1], 'ro',  label='Current Location')  # Plot the current point as a thick red dot
        
        
        # self.ax2.plot(self.s_left, self.d_left, color='orange', label='Left Boundary')  
        # self.ax2.plot(self.s_right, self.d_right, color='tan', label='Right Boundary')  
        self.ax2.scatter(self.reference_s, self.ref_left_boundary_d, color='orange', s=5, label='Left Boundary (Ref)')  
        self.ax2.scatter(self.reference_s, self.ref_right_boundary_d, color='tan', s=5, label='Right Boundary (Ref)')  
        
        zoom_range = self.frenet_zoom
        if self.past_s and not np.isnan(self.past_s[-1]) and not np.isinf(self.past_s[-1]):
            self.ax2.set_xlim(self.past_s[-1] - zoom_range/4, self.past_s[-1] + zoom_range)
        self.ax2.set_ylim(-zoom_range,  zoom_range)
        

        #  print lattice graph
        for k,v in self.lattice_graph.items():
            self.ax2.plot(v.ts, v.td,"m--")
            self.ax1.plot(v.tx, v.ty,"m--")
            self.ax1.plot(v.tx[-1], v.ty[-1], 'go')
            self.ax2.plot(v.ts[-1],v.td[-1], 'go')
            for v.next_edge in v.next_edges:
                self.ax2.plot(v.next_edge.ts, v.next_edge.td, 'g--', alpha=0.5)
                self.ax1.plot(v.next_edge.tx, v.next_edge.ty, 'g--', alpha=0.5)
                self.ax1.plot(v.next_edge.tx[-1], v.next_edge.ty[-1], 'yo')
                self.ax2.plot(v.next_edge.ts[-1], v.next_edge.td[-1], 'yo')
            # print next edges

        self.ax2.legend(loc='upper left')
        self.ax2.set_title('Frenet Coordinate')
        
        
        self.ax1.legend(loc='upper left')
        # plt.pause(pause_interval)

    def plt_show(self):
        plt.show()
    
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

if __name__ == '__main__':
    import visualizer 
    visualizer.main()
    
