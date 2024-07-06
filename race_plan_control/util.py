from plan.planner import Planner
import time
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot(pl: Planner, cn = None, frenet_zoom = 15, xy_zoom = 30):
    if len(pl.xdata) == 0:
        print("No data to plot")
        return
    
    ax1.clear()
    ax2.clear()
    

    # Plot track boundaries in black

    ax1.plot(pl.left_x,pl.left_y, color='orange', label='Left Boundary')  # Change color and label as needed
    ax1.plot(pl.right_x, pl.right_y, color='tan', label='Right Boundary')  # Change color and label as needed
    

    # Plot the reference path in blue
    ax1.plot(pl.reference_x, pl.reference_y, 'b', label="Reference Trajectory")  # reference path in blue
    # self.ax1.scatter(self.reference_x, self.reference_y, color="blue", label="Reference Trajectory")  # reference path in blue
    
    # For Zoom in
    if xy_zoom is not None:
        range = xy_zoom
        ax1.set_xlim(pl.xdata[-1] - 2*range, pl.xdata[-1] + 2*range)
        ax1.set_ylim(pl.ydata[-1] - range, pl.ydata[-1] + range)
    
    if pl.past_s and not np.isnan(pl.past_s[-1]) and not np.isinf(pl.past_s[-1]):
        ax2.set_xlim(pl.past_s[-1] - 2*frenet_zoom/4 , pl.past_s[-1] + 4*frenet_zoom)
    ax2.set_ylim(-frenet_zoom,  frenet_zoom)
    
    ax1.plot(pl.xdata, pl.ydata, 'r-', label='Past Locations')  # Plot all points in red
    ax1.plot(pl.xdata[-100:], pl.ydata[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
    ax1.plot(pl.xdata[-1], pl.ydata[-1], 'ro', markersize=10, label='Current Location')  

    if pl.selected_edge is not None:
        x,y = pl.selected_edge.get_current_xy() 
        s,d = pl.selected_edge.get_current_sd()
        ax1.plot(x,y, 'bo', markersize=7, label='Planned Location')  
        ax2.plot(s,d, 'bo', markersize=9, label='Planned Location')  
        
    
    if pl.race_trajectory.next_wp is not None: 
        ax1.plot(pl.reference_x[pl.race_trajectory.next_wp], pl.reference_y[pl.race_trajectory.next_wp], 'gx', markersize=10, label='Next WP')  
        ax2.plot(pl.reference_s[pl.race_trajectory.next_wp], pl.reference_d[pl.race_trajectory.next_wp], 'gx', markersize=10, label='Next WP')  


    # Use ax2 for the Frenet coordinates
    
    # self.ax2.axhline(0, color='blue', linewidth=0.5, label='Reference Path') 
    ax2.scatter(pl.reference_s, pl.reference_d, s=5, alpha=.5, color="blue", label="Reference Trajectory")  # reference path in blue
    

    current_time = time.time()
    if pl.prev_time is not None:
        pl.x_vel = (pl.past_s[-1] - pl.past_s[-2]) / (current_time - pl.prev_time)
        pl.y_vel = (pl.past_d[-1] - pl.past_d[-2]) / (current_time - pl.prev_time)
        ax2.text(pl.past_s[-1]-1.1, pl.past_d[-1]+1, f'X Velocity: {pl.x_vel:.2f}\nY Velocity: {pl.y_vel:.2f}', verticalalignment='bottom', fontsize=9)
        ax2.text(pl.past_s[-1]-1.1, pl.past_d[-1]-1, f'Current d: {pl.past_d[-1]:.2f}\nMSE: {pl.mse:.2f}',
                    verticalalignment='top', fontsize=12)
    pl.prev_time = current_time
    
    if not len(pl.past_s)==0:
        ax2.plot(pl.past_s, pl.past_d, 'r-', label='Past Locations')  # Plot all points in red
        ax2.plot(pl.past_s[-100:], pl.past_d[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
        ax2.plot(pl.past_s[-1], pl.past_d[-1], 'ro',  label='Current Location')  # Plot the current point as a thick red dot
    
    
    # self.ax2.plot(self.s_left, self.d_left, color='orange', label='Left Boundary')  
    # self.ax2.plot(self.s_right, self.d_right, color='tan', label='Right Boundary')  
    ax2.scatter(pl.reference_s, pl.ref_left_boundary_d, color='orange', s=5, label='Left Boundary (Ref)')  
    ax2.scatter(pl.reference_s, pl.ref_right_boundary_d, color='tan', s=5, label='Right Boundary (Ref)')  
    
    

    #  print lattice graph
    for k,v in pl.lattice_graph.items():
        ax2.plot(v.ts, v.td,"m--")
        ax1.plot(v.tx, v.ty,"m--")
        ax1.plot(v.tx[-1], v.ty[-1], 'go')
        ax2.plot(v.ts[-1],v.td[-1], 'go')
        for v.next_edge in v.next_edges:
            ax2.plot(v.next_edge.ts, v.next_edge.td, 'g--', alpha=0.5)
            ax1.plot(v.next_edge.tx, v.next_edge.ty, 'g--', alpha=0.5)
            ax1.plot(v.next_edge.tx[-1], v.next_edge.ty[-1], 'yo')
            ax2.plot(v.next_edge.ts[-1], v.next_edge.td[-1], 'yo')
        # print next edges

    # ax2.legend(loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, -.1), ncol=4, borderaxespad=0.)

    ax2.set_title('Frenet Coordinate')
    
    
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')



fig, (ax1, ax2) = plt.subplots(2, 1)


if __name__ == "__main__":
    import visualizer 
    sys.path.append("..")
    visualizer.main()