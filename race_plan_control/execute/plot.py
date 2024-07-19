import race_plan_control.main as main
from race_plan_control.plan.planner import Planner
from race_plan_control.execute.executer import Executer

import numpy as np
import matplotlib.pyplot as plt



def plot(pl: Planner, exec: Executer = None, aspect_ratio=4, frenet_zoom = 15, xy_zoom = 30):
    ax1.clear()
    ax2.clear()
    

    # Plot track boundaries 
    ax1.plot(pl.left_x,pl.left_y, color='orange', label='Left Boundary', linewidth=2)  # Change color and label as needed
    ax1.plot(pl.right_x, pl.right_y, color='tan', label='Right Boundary', linewidth=2)  # Change color and label as needed
    
    # Plot the reference path in blue
    ax1.plot(pl.global_trajectory.reference_x, pl.global_trajectory.reference_y, 'gray', label="Reference Trajectory", linewidth=2)  # reference path in gray
    ax2.scatter(pl.global_trajectory.reference_s, pl.global_trajectory.reference_d, s=5, alpha=.5, color="gray", label="Reference Trajectory")  # reference path in blue

    
    
    # ax1.plot(pl.xdata, pl.ydata, 'r-', label='Past Locations', linewidth=2)  # Plot all points in red
    ax1.plot(pl.xdata[-100:], pl.ydata[-100:], 'g-', label='Last 100 Locations', linewidth=2)  # Plot the last 100 points in green
    ax1.plot(pl.xdata[-1], pl.ydata[-1], 'ro', markersize=10, label='Planner Location')  
    # For Zoom in
    if xy_zoom is not None:
        ax1.set_xlim(pl.xdata[-1] - xy_zoom, pl.xdata[-1] + xy_zoom)
        ax1.set_ylim(pl.ydata[-1] - xy_zoom/aspect_ratio/2, pl.ydata[-1] + xy_zoom/aspect_ratio/2)
    
    if pl.past_s and not np.isnan(pl.past_s[-1]) and not np.isinf(pl.past_s[-1]):
        ax2.set_xlim(pl.past_s[-1] - frenet_zoom/2 ,   pl.past_s[-1] + 1.5*frenet_zoom)
        ax2.set_ylim(-frenet_zoom/aspect_ratio/2,  frenet_zoom/aspect_ratio/2)


#---------------------------
# Plot Global Plan ---------
#---------------------------
    if pl.global_trajectory.next_wp is not None: 
        ax1.plot(pl.global_trajectory.reference_x[pl.global_trajectory.current_wp],
                    pl.global_trajectory.reference_y[pl.global_trajectory.current_wp], 'g', markersize=13, label='G WP: Curent', marker='o', fillstyle='none')
        ax2.plot(pl.global_trajectory.reference_s[pl.global_trajectory.current_wp],
                    pl.global_trajectory.reference_d[pl.global_trajectory.current_wp], 'g', markersize=13, label='G WP: Curent', marker='o', fillstyle='none')  

        ax1.plot(pl.global_trajectory.reference_x[pl.global_trajectory.next_wp],
                    pl.global_trajectory.reference_y[pl.global_trajectory.next_wp], 'gx', markersize=13, label='G WP: Next')  
        ax2.plot(pl.global_trajectory.reference_s[pl.global_trajectory.next_wp],
                    pl.global_trajectory.reference_d[pl.global_trajectory.next_wp], 'gx', markersize=13, label='G WP: Next')  
            
    if not len(pl.past_s)==0:
        ax2.plot(pl.past_s[-100:], pl.past_d[-100:], 'g-', label='Last 100 Locations')  # Plot the last 100 points in green
        ax2.plot(pl.past_s[-1], pl.past_d[-1], 'ro',  label='Current Location')  # Plot the current point as a thick red dot
        
    ax2.scatter(pl.global_trajectory.reference_s, pl.ref_left_boundary_d, color='orange', s=5, label='Left Boundary (Ref)')  
    ax2.scatter(pl.global_trajectory.reference_s, pl.ref_right_boundary_d, color='tan', s=5, label='Right Boundary (Ref)')  
    
    



    # -------------------------------
    # -Plot local Plan---------------
    # -------------------------------

    #  print lattice graph
    for k,v in pl.lattice_graph.items():
        ax2.plot(v.ts, v.td,"b--",alpha=0.6)
        ax1.plot(v.tx, v.ty,"b--", alpha=0.6)
        ax1.plot(v.tx[-1], v.ty[-1], 'bo', alpha=0.6)
        ax2.plot(v.ts[-1],v.td[-1], 'bo', alpha=0.6)
        for v.next_edge in v.next_edges:
            ax2.plot(v.next_edge.ts, v.next_edge.td, 'b--', alpha=0.5)
            ax1.plot(v.next_edge.tx, v.next_edge.ty, 'b--', alpha=0.5)
            ax1.plot(v.next_edge.tx[-1], v.next_edge.ty[-1], 'bo', alpha=0.5)
            ax2.plot(v.next_edge.ts[-1], v.next_edge.td[-1], 'bo', alpha=0.5)

    # printing local graph
    if pl.selected_edge is not None:


        x,y = pl.selected_edge.local_trajectory.get_xy()
        s = pl.selected_edge.ts[pl.selected_edge.local_trajectory.current_wp]
        d = pl.selected_edge.td[pl.selected_edge.local_trajectory.current_wp]

        x_n, y_n = pl.selected_edge.local_trajectory.get_xy(pl.selected_edge.local_trajectory.next_wp)
        s_n = pl.selected_edge.ts[pl.selected_edge.local_trajectory.next_wp]
        d_n = pl.selected_edge.td[pl.selected_edge.local_trajectory.next_wp]
        



        ax1.plot(x,y, 'bo', markersize=15, label='L WP: Current', fillstyle='none')  
        ax2.plot(s,d, 'bo', markersize=15, label='L WP: Current', fillstyle='none')  
        
        ax1.plot(x_n,y_n, 'bx', markersize=15, label='L WP: Next', fillstyle='none')  
        ax2.plot(s_n,d_n, 'bx', markersize=15, label='L WP: Next', fillstyle='none')  
        
        v = pl.selected_edge
        ax1.plot(v.tx, v.ty, 'r-', label="Selected Edge", alpha = 0.6, linewidth=6)
        ax2.plot(v.ts, v.td, 'r-', label="Selected Edge", alpha = 0.6, linewidth=6)

        if v.selected_next_edge is not None:
            ax1.plot(v.selected_next_edge.tx, v.selected_next_edge.ty, 'r-', label='Next Edge', alpha = 0.3, linewidth=6)
            ax2.plot(v.selected_next_edge.ts, v.selected_next_edge.td, 'r-', label='Next Edge', alpha = 0.3, linewidth=6)

            
    if exec is not None:
        # Plot the car
        car_L_f = exec.state.L_f
        car_L_r = exec.state.length - car_L_f

        car_x_front = exec.state.x + car_L_f * np.cos(exec.state.theta)
        car_y_front = exec.state.y + car_L_f * np.sin(exec.state.theta)
        car_x_rear = exec.state.x - car_L_r * np.cos(exec.state.theta)
        car_y_rear = exec.state.y - car_L_r * np.sin(exec.state.theta)
        ax1.plot([car_x_front, car_x_rear], [car_y_front, car_y_rear], 'k-', label='Car Heading')
        ax1.plot(exec.state.x, exec.state.y, 'ko', markersize=7, label='Car Location')


    # ax2.legend(loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, -.1), ncol=5, borderaxespad=0.)

    ax2.set_title('Frenet Coordinate')
    
    
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    # plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=.99, bottom=0.1)



fig, (ax1, ax2) = plt.subplots(2, 1)
if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()