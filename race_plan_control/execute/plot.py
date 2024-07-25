import race_plan_control.main as main
from race_plan_control.plan.planner import Planner
from race_plan_control.execute.executer import Executer

import numpy as np
import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(2, 1)
left_boundry_x1, = ax1.plot([],[], color='orange', label='Left Boundary', linewidth=2)  # Change color and label as needed
right_boundry_x1, = ax1.plot([],[], color='tan', label='Right Boundary', linewidth=2)  # Change color and label as needed
left_boundry_ax2 = ax2.scatter([],[], color='orange', s=5, label='Left Boundary (Ref)')  
right_boundry_ax2 = ax2.scatter([],[], color='tan', s=5, label='Right Boundary (Ref)')  

reference_trajectory_ax1, = ax1.plot([],[], 'gray', label="Reference Trajectory", linewidth=2)  # reference path in gray
reference_trajectory_ax2 = ax2.scatter([],[], s=5, alpha=.5, color="gray", label="Reference Trajectory")  # reference path in blue

last_100_locs_ax1, = ax1.plot([],[], 'g-', label='Last 100 Locations', linewidth=2)  # Plot the last 100 points in green
planner_loc_ax1, = ax1.plot([],[], 'ro', markersize=10, label='Planner Location')

last_100_locs_ax2, = ax2.plot([],[], 'g-', label='Last 100 Locations', linewidth=2)  # Plot the last 100 points in green
planner_loc_ax2, = ax2.plot([],[], 'ro', markersize=10, label='Planner Location')


g_wp_current_ax1, = ax1.plot([],[], 'g', markersize=13, label='G WP: Curent', marker='o', fillstyle='none')
g_wp_current_ax2,= ax2.plot([],[], 'g', markersize=13, label='G WP: Curent', marker='o', fillstyle='none')  

g_wp_next_ax1, = ax1.plot([],[], 'gx', markersize=13, label='G WP: Next')  
g_wp_next_ax2, = ax2.plot([],[], 'gx', markersize=13, label='G WP: Next')  


# Local Plan initialization
# Initialize lattice graph plots
lattice_graph_plots_ax1 = []
lattice_graph_plots_ax2 = []
lattice_graph_endpoints_ax1 = []
lattice_graph_endpoints_ax2 = []

# Assuming a maximum number of lattice graph edges
max_lattice_edges = 100

for _ in range(max_lattice_edges):
    line_ax1, = ax1.plot([], [], "b--", alpha=0.6)
    line_ax2, = ax2.plot([], [], "b--", alpha=0.6)
    endpoint_ax1, = ax1.plot([], [], 'bo', alpha=0.6)
    endpoint_ax2, = ax2.plot([], [], 'bo', alpha=0.6)
    lattice_graph_plots_ax1.append(line_ax1)
    lattice_graph_plots_ax2.append(line_ax2)
    lattice_graph_endpoints_ax1.append(endpoint_ax1)
    lattice_graph_endpoints_ax2.append(endpoint_ax2)

selected_edge_plot_ax1, = ax1.plot([], [], 'r-', label="Selected Edge", alpha=0.6, linewidth=6)
selected_edge_plot_ax2, = ax2.plot([], [], 'r-', label="Selected Edge", alpha=0.6, linewidth=6)
selected_next_edge_plot_ax1, = ax1.plot([], [], 'r-', label='Next Edge', alpha=0.3, linewidth=6)
selected_next_edge_plot_ax2, = ax2.plot([], [], 'r-', label='Next Edge', alpha=0.3, linewidth=6)
current_wp_plot_ax1, = ax1.plot([], [], 'bo', markersize=15, label='L WP: Current', fillstyle='none')
current_wp_plot_ax2, = ax2.plot([], [], 'bo', markersize=15, label='L WP: Current', fillstyle='none')
next_wp_plot_ax1, = ax1.plot([], [], 'bx', markersize=15, label='L WP: Next', fillstyle='none')
next_wp_plot_ax2, = ax2.plot([], [], 'bx', markersize=15, label='L WP: Next', fillstyle='none')

# Control Init
car_heading_plot, = ax1.plot([], [], 'k-', label='Car Heading')
car_location_plot, = ax1.plot([], [], 'ko', markersize=7, label='Car Location')

ax2.set_title('Frenet Coordinate')
ax2.legend(loc='upper left', bbox_to_anchor=(0, -.1), ncol=5, borderaxespad=0.)
ax1.legend(loc='upper left')
ax1.set_aspect('equal')
ax2.set_aspect('equal')

fig.subplots_adjust(left=0, right=1, top=.99, bottom=0.1)

initialized = False

bg = fig.canvas.copy_from_bbox(fig.bbox)
fig.canvas.blit(fig.bbox)

def plot(pl: Planner, exec: Executer = None, aspect_ratio=4, frenet_zoom = 15, xy_zoom = 30, plot_global_plan = False, plot_local_plan = False, plot_control = False):
    global initialized
    fig.canvas.restore_region(bg)
    if plot_global_plan and not initialized:
        # Plot track boundaries 
        left_boundry_x1.set_data(pl.left_x, pl.left_y)
        right_boundry_x1.set_data(pl.right_x, pl.right_y)
        left_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.reference_s, pl.ref_left_boundary_d])
        right_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.reference_s, pl.ref_right_boundary_d]) 
        # Plot the reference path 
        reference_trajectory_ax1.set_data(pl.global_trajectory.reference_x, pl.global_trajectory.reference_y)
        reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.reference_s, pl.global_trajectory.reference_d])
        
        initialized = True
    
    
    
    # For Zoom in
    if xy_zoom is not None:
        ax1.set_xlim(pl.xdata[-1] - xy_zoom, pl.xdata[-1] + xy_zoom)
        ax1.set_ylim(pl.ydata[-1] - xy_zoom/aspect_ratio/2, pl.ydata[-1] + xy_zoom/aspect_ratio/2)
    if frenet_zoom is not None:
        ax2.set_xlim(pl.past_s[-1] - frenet_zoom/2 ,   pl.past_s[-1] + 1.5*frenet_zoom)
        ax2.set_ylim(-frenet_zoom/aspect_ratio/2,  frenet_zoom/aspect_ratio/2)


    #---------------------------
    # Plot Global Plan ---------
    #---------------------------
    if plot_global_plan:
        if pl.global_trajectory.next_wp is not None: 
            g_wp_current_ax1.set_data([pl.global_trajectory.reference_x[pl.global_trajectory.current_wp]], [pl.global_trajectory.reference_y[pl.global_trajectory.current_wp]])
            g_wp_current_ax2.set_data([pl.global_trajectory.reference_s[pl.global_trajectory.current_wp]], [pl.global_trajectory.reference_d[pl.global_trajectory.current_wp]])
            g_wp_next_ax1.set_data([pl.global_trajectory.reference_x[pl.global_trajectory.next_wp]], [pl.global_trajectory.reference_y[pl.global_trajectory.next_wp]])
            g_wp_next_ax2.set_data([pl.global_trajectory.reference_s[pl.global_trajectory.next_wp]], [pl.global_trajectory.reference_d[pl.global_trajectory.next_wp]])
        
        last_100_locs_ax1.set_data(pl.xdata[-100:], pl.ydata[-100:])
        planner_loc_ax1.set_data([pl.xdata[-1]], [pl.ydata[-1]])  
        
        last_100_locs_ax2.set_data(pl.past_s[-100:], pl.past_d[-100:]) 
        planner_loc_ax2.set_data([pl.past_s[-1]], [pl.past_d[-1]])  
            



    # -------------------------------
    # -Plot local Plan---------------
    # -------------------------------

    if plot_local_plan:
        _plot_lattice_graph(pl)
        if pl.selected_edge is not None:
            x, y = pl.selected_edge.local_trajectory.get_xy()
            s = pl.selected_edge.ts[pl.selected_edge.local_trajectory.current_wp]
            d = pl.selected_edge.td[pl.selected_edge.local_trajectory.current_wp]
        
            x_n, y_n = pl.selected_edge.local_trajectory.get_xy(pl.selected_edge.local_trajectory.next_wp)
            s_n = pl.selected_edge.ts[pl.selected_edge.local_trajectory.next_wp]
            d_n = pl.selected_edge.td[pl.selected_edge.local_trajectory.next_wp]
        
            current_wp_plot_ax1.set_data([x], [y])
            current_wp_plot_ax2.set_data([s], [d])
            
            next_wp_plot_ax1.set_data([x_n], [y_n])
            next_wp_plot_ax2.set_data([s_n], [d_n])
            
            v = pl.selected_edge
            selected_edge_plot_ax1.set_data(v.tx, v.ty)
            selected_edge_plot_ax2.set_data(v.ts, v.td)
        
            if v.selected_next_edge is not None:
                selected_next_edge_plot_ax1.set_data(v.selected_next_edge.tx, v.selected_next_edge.ty)
                selected_next_edge_plot_ax2.set_data(v.selected_next_edge.ts, v.selected_next_edge.td)

            
    if plot_control and exec is not None:
        car_L_f = exec.state.L_f
        car_L_r = exec.state.length - car_L_f
    
        car_x_front = exec.state.x + car_L_f * np.cos(exec.state.theta)
        car_y_front = exec.state.y + car_L_f * np.sin(exec.state.theta)
        car_x_rear = exec.state.x - car_L_r * np.cos(exec.state.theta)
        car_y_rear = exec.state.y - car_L_r * np.sin(exec.state.theta)
    
        car_heading_plot.set_data([car_x_front, car_x_rear], [car_y_front, car_y_rear])
        car_location_plot.set_data([exec.state.x], [exec.state.y])

    _redraw_plots()
    # fig.canvas.flush_events()

def _redraw_plots():
    # Redraw only the updated parts
    ax1.draw_artist(ax1.patch)
    ax2.draw_artist(ax2.patch)
    for line in ax1.lines + ax2.lines:
        ax1.draw_artist(line)
        ax2.draw_artist(line)

    fig.canvas.blit(ax1.bbox)
    fig.canvas.blit(ax2.bbox)
    # fig.canvas.flush_events()


def _plot_lattice_graph(pl):
    edge_index = 0
    for k, v in pl.lattice_graph.items():
        if edge_index < max_lattice_edges:
            lattice_graph_plots_ax1[edge_index].set_data(v.tx, v.ty)
            lattice_graph_plots_ax2[edge_index].set_data(v.ts, v.td)
            lattice_graph_endpoints_ax1[edge_index].set_data([v.tx[-1]], [v.ty[-1]])
            lattice_graph_endpoints_ax2[edge_index].set_data([v.ts[-1]], [v.td[-1]])
            edge_index += 1
        for next_edge in v.next_edges:
            if edge_index < max_lattice_edges:
                lattice_graph_plots_ax1[edge_index].set_data(next_edge.tx, next_edge.ty)
                lattice_graph_plots_ax2[edge_index].set_data(next_edge.ts, next_edge.td)
                lattice_graph_endpoints_ax1[edge_index].set_data([next_edge.tx[-1]], [next_edge.ty[-1]])
                lattice_graph_endpoints_ax2[edge_index].set_data([next_edge.ts[-1]], [next_edge.td[-1]])
                edge_index += 1


if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()