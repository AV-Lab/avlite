import rclpy
from rclpy.node import Node
from a2rl_bs_msgs.msg import Localization, EgoState
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import inset_locator
import json
import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial
import math

import yaml


import matplotlib
import math
import util as u

matplotlib.use("TkAgg")


class local_planner(Node):
    def __init__(self, path_to_track, track_boundaries_path, frenet_zoom=15, xy_zoom=15):
        super().__init__('local_planner')
        self.frenet_zoom = frenet_zoom
        self.xy_zoom = xy_zoom
        
        self.create_subscription(Localization,'/a2rl/observer/ego_loc', self.loc_callback,10)
        self.create_subscription(EgoState,'/a2rl/observer/ego_state', self.state_callback,10)

        with open(path_to_track, 'r') as f:
            track_data = json.load(f)

        # with open(track_boundaries_path, 'r') as f:
            # track_boundaries = json.load(f)


        self.xdata, self.ydata = [], []
        
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

        self.d = []
        self.s = []
        self.mse = 0


        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        # self.ax3 = inset_locator.inset_axes(self.ax1, width="30%", height="30%", loc="upper right")

        self.create_timer(0.1, self.plot)


    def state_callback(self, msg):
        # self.get_logger().info('Velocity x: %s' % str(msg.velocity.x))
        self.x_vel = msg.velocity.x
        self.y_vel = msg.velocity.y


        
    def loc_callback(self, msg):
        x_current = msg.position.x
        y_current = msg.position.y
        self.xdata.append(msg.position.x)
        self.ydata.append(msg.position.y)
        
        self.tj.update_waypoint(x_current, y_current)
        
        #### Frenet Coordinates
        s_, d_= self.tj.convert_to_frenet([(msg.position.x, msg.position.y)])
        self.d.append(d_[0])
        self.s.append(s_[0])
        
        if len(self.d)>0:
            d_mean = sum(self.d) / len(self.d) 
            self.mse = sum((di - d_mean)**2 for di in self.d) / len(self.d) 
        
    def plot(self):
        if len(self.xdata) == 0:
            print("No data to plot")
            return
        self.ax1.clear()
        self.ax2.clear()

        # Plot track boundaries in black

        self.ax1.plot(self.left_x,self.left_y, color='orange', label='Left Boundary')  # Change color and label as needed
        self.ax1.plot(self.right_x, self.right_y, color='tan', label='Right Boundary')  # Change color and label as needed
        
        # self.ax1.scatter(self.reference_s, self.ref_right_boundary_d, color='tan', s=5, label='Right Boundary (Ref)')  


        # Plot the reference path in blue
        self.ax1.plot(self.reference_x, self.reference_y, 'b', label="Reference Trajectory")  # reference path in blue
        
        # For Zoom in
        if self.xy_zoom is not None:
            range = self.xy_zoom
            self.ax1.set_xlim(self.xdata[-1] - range, self.xdata[-1] + range)
            self.ax1.set_ylim(self.ydata[-1] - range, self.ydata[-1] + range)
        
        self.ax1.plot(self.xdata, self.ydata, 'r-', label='Past Locations')  # Plot all points in red
        self.ax1.plot(self.xdata[-100:], self.ydata[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
        self.ax1.plot(self.xdata[-1], self.ydata[-1], 'ro', markersize=10, label='Current Location')  
        
        
        if self.tj.next_wp is not None: 
            self.ax1.plot(self.reference_x[self.tj.next_wp], self.reference_y[self.tj.next_wp], 'gx', markersize=10, label='Next WP')  
            self.ax2.plot(self.reference_s[self.tj.next_wp], self.reference_d[self.tj.next_wp], 'gx', markersize=10, label='Next WP')  
        

        # Use ax2 for the Frenet coordinates
        
        self.ax2.axhline(0, color='blue', linewidth=0.5, label='Reference Path') 
       
        #self.ax2.plot(self.s_ref, self.d_ref, color='Blue', linestyle='--', label='Reference Path')

        # Display the current d value and its squared error
        self.ax2.text(self.s[-1]+1, self.d[-1], f'Current d: {self.d[-1]:.2f}\nMSE: {self.mse:.2f}',
                    verticalalignment='top', fontsize=12)
        self.ax2.text(self.s[-1]+1, self.d[-1], f'X Velocity: {self.x_vel:.2f}\nY Velocity: {self.y_vel:.2f}', verticalalignment='bottom', fontsize=9)
       
        self.ax2.plot(self.s, self.d, 'r-', label='Past Locations')  # Plot all points in red
        self.ax2.plot(self.s[-100:], self.d[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
        self.ax2.plot(self.s[-1], self.d[-1], 'ro',  label='Current Location')  # Plot the current point as a thick red dot
        
        
        # self.ax2.plot(self.s_left, self.d_left, color='orange', label='Left Boundary')  
        # self.ax2.plot(self.s_right, self.d_right, color='tan', label='Right Boundary')  
        self.ax2.scatter(self.reference_s, self.ref_left_boundary_d, color='orange', s=5, label='Left Boundary (Ref)')  
        self.ax2.scatter(self.reference_s, self.ref_right_boundary_d, color='tan', s=5, label='Right Boundary (Ref)')  
        
        zoom_range = self.frenet_zoom
        if self.s and not np.isnan(self.s[-1]) and not np.isinf(self.s[-1]):
            self.ax2.set_xlim(self.s[-1] - zoom_range, self.s[-1] + zoom_range)
        self.ax2.set_ylim(-zoom_range,  zoom_range)
        


        s_ = self.s[-1] +5
        d_ = self.d[-1] +2
        x_, y_ = self.tj.getXY(s_, d_)
        # print(f'x: {self.xdata[-1]}, y: {self.ydata[-1]}')
        # print(f'x_: {x_}, y_: {y_}')
        self.ax1.plot(x_,y_, 'go',  label='Goal Location')  # Plot the current point as a thick red dot


        self.ax2.plot(s_,d_, 'go',  label='Goal Location')  # Plot the current point as a thick red dot
        ts,td,tx,ty= self.tj.generate_trajectory(self.s[-1], s_, self.d[-1], d_)
        self.ax2.plot(ts, td, 'm--', label='Trajectory')  # Plot the trajectory in purple dashed line

        self.ax1.plot(tx, ty, 'm--', label='Trajectory')  # Plot the trajectory in purple dashed line

        self.ax2.legend(loc='upper left')
        self.ax2.set_title('Frenet Coordinate')
        
        
        self.ax1.legend(loc='upper left')
        # plt.draw()
        plt.pause(0.01)
        
    

def main(args=None):
    rclpy.init(args=args)
    
    track_boundaries_path= "/home/mkhonji/workspaces/A2RL_Integration/src/1_dashboards/resource/yasmarina.track.json"

    config_file = "/home/mkhonji/workspaces/A2RL_Integration/config/config.yaml"
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        path_to_track = config_data["path_to_track"]
    
    plot_subscriber = local_planner(path_to_track, track_boundaries_path)

    rclpy.spin(plot_subscriber)
    
    plot_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
