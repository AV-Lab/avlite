import race_plan_control.main as main
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

last_locs_ax1, = ax1.plot([],[], 'g-', label='Last 100 Locations', linewidth=2)  # Plot the last 100 points in green
planner_loc_ax1, = ax1.plot([],[], 'ro', markersize=10, label='Planner Location')

last_locs_ax2, = ax2.plot([],[], 'g-', label='Last 100 Locations', linewidth=2)  # Plot the last 100 points in green
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


selected_edge_plot_ax1, = ax1.plot([], [], 'r-', label="Selected Edge", alpha=0.6, linewidth=6)
selected_edge_plot_ax2, = ax2.plot([], [], 'r-', label="Selected Edge", alpha=0.6, linewidth=6)
selected_next_edge_plot_ax1, = ax1.plot([], [], 'r-', label='Next Edge', alpha=0.3, linewidth=6)
selected_next_edge_plot_ax2, = ax2.plot([], [], 'r-', label='Next Edge', alpha=0.3, linewidth=6)
current_wp_plot_ax1, = ax1.plot([], [], 'bo', markersize=15, label='L WP: Current', fillstyle='none')
current_wp_plot_ax2, = ax2.plot([], [], 'bo', markersize=15, label='L WP: Current', fillstyle='none')
next_wp_plot_ax1, = ax1.plot([], [], 'bx', markersize=15, label='L WP: Next', fillstyle='none')
next_wp_plot_ax2, = ax2.plot([], [], 'bx', markersize=15, label='L WP: Next', fillstyle='none')

# State Init
car_heading_plot, = ax1.plot([], [], 'k-', label='Car Heading')
car_location_plot, = ax1.plot([], [], 'ko', markersize=7, label='Car Location')
car_rect = plt.Rectangle((0,0), 0, 0, angle=0, edgecolor='r', facecolor='none')
ax1.add_patch(car_rect)

# Assuming a maximum number of lattice graph edges
max_lattice_edges = 10

for _ in range(max_lattice_edges):
    line_ax1, = ax1.plot([], [], "b--", alpha=0.6)
    line_ax2, = ax2.plot([], [], "b--", alpha=0.6)
    endpoint_ax1, = ax1.plot([], [], 'bo', alpha=0.6)
    endpoint_ax2, = ax2.plot([], [], 'bo', alpha=0.6)
    lattice_graph_plots_ax1.append(line_ax1)
    lattice_graph_plots_ax2.append(line_ax2)
    lattice_graph_endpoints_ax1.append(endpoint_ax1)
    lattice_graph_endpoints_ax2.append(endpoint_ax2)

ax2.set_title('Frenet Coordinate')
ax1.set_aspect('equal')
ax2.set_aspect('equal')

# ax1.legend(loc='upper left')
# legend_ax1 = set(ax1.get_legend().get_lines())
# ax2.legend(loc='upper left', bbox_to_anchor=(0, -.1), ncol=5, borderaxespad=0.)
# ax2.legend(loc='lower center', ncol=5, borderaxespad=0.)
# legend_ax2 = {} # set(ax2.get_legend().get_lines())

legend_ax = fig.add_axes([0.0, -.013, 1, 0.1])
legend_ax.legend(*ax1.get_legend_handles_labels(), loc='center', ncol=5, borderaxespad=0.)
legend_ax.axis('off')
# fig.canvas.blit(legend_ax.bbox)

fig.subplots_adjust(left=0, right=1, top=.99, bottom=0.1)



def plot(exec: Executer, aspect_ratio=4, frenet_zoom = 15, xy_zoom = 30, show_legend = True, plot_last_pts = True,
           plot_global_plan = True,   plot_local_plan = True, plot_local_lattice = True, plot_state = True,num_plot_last_pts = 100):
    
    legend_ax.set_visible(show_legend)
    
    
    # For Zoom in
    if xy_zoom is not None:
        ax1.set_xlim(exec.pl.xdata[-1] - xy_zoom, exec.pl.xdata[-1] + xy_zoom)
        ax1.set_ylim(exec.pl.ydata[-1] - xy_zoom/aspect_ratio/2, exec.pl.ydata[-1] + xy_zoom/aspect_ratio/2)
    if frenet_zoom is not None:
        ax2.set_xlim(exec.pl.past_s[-1] - frenet_zoom/2 ,   exec.pl.past_s[-1] + 1.5*frenet_zoom)
        ax2.set_ylim(-frenet_zoom/aspect_ratio/2,  frenet_zoom/aspect_ratio/2)


    if plot_last_pts and num_plot_last_pts > 0:
        last_locs_ax1.set_data(exec.pl.xdata[-num_plot_last_pts:], exec.pl.ydata[-num_plot_last_pts:])
        planner_loc_ax1.set_data([exec.pl.xdata[-1]], [exec.pl.ydata[-1]])  
        
        last_locs_ax2.set_data(exec.pl.past_s[-num_plot_last_pts:], exec.pl.past_d[-num_plot_last_pts:]) 
        planner_loc_ax2.set_data([exec.pl.past_s[-1]], [exec.pl.past_d[-1]])  
    else:
        last_locs_ax1.set_data([], [])
        planner_loc_ax1.set_data([], [])
        last_locs_ax2.set_data([], [])
        planner_loc_ax2.set_data([], [])
            
    # update global Plan
    update_global_plan_plots(exec.pl, plot_global_plan) 

    # update local plan
    update_lattice_graph_plots(exec.pl, plot_local_lattice)
    update_local_plan_plots(exec.pl, plot_local_plan)

    # update state 
    update_state_plots(exec.state, plot_state)

    redraw_plots()

def redraw_plots():
    # Redraw only the updated parts
    # Draw the background patches
    ax1.draw_artist(ax1.patch)
    ax2.draw_artist(ax2.patch)
    
    # Draw only the lines, excluding the legend
    for line in ax1.lines:
        # if line not in legend_ax1:
        ax1.draw_artist(line)
    for line in ax2.lines:
        # if line not in legend_ax2:
        ax2.draw_artist(line)
    
    # Update the canvas
    fig.canvas.blit(ax1.bbox)
    fig.canvas.blit(ax2.bbox)
    # fig.canvas.flush_events()

initialized = False
toggle_plot = False
def update_global_plan_plots(pl, show_plot = True):
        global initialized
        global toggle_plot
        if not initialized:
            # Plot track boundaries 
            left_boundry_x1.set_data(pl.left_x, pl.left_y)
            right_boundry_x1.set_data(pl.right_x, pl.right_y)
            left_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.ref_left_boundary_d])
            right_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.ref_right_boundary_d]) 
            # Plot the reference path 
            reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)
            reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.global_trajectory.path_d])
            initialized = True
        if not show_plot:
            g_wp_current_ax1.set_data([], [])
            g_wp_current_ax2.set_data([], [])
            g_wp_next_ax1.set_data([], [])
            g_wp_next_ax2.set_data([], [])
            reference_trajectory_ax1.set_data([],[])
            reference_trajectory_ax2.set_offsets(np.c_[[],[]])
            toggle_plot = True
            return 
        elif initialized and toggle_plot:
            reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)
            reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.global_trajectory.path_d])
            toggle_plot = False
    
        if pl.global_trajectory.next_wp is not None: 
            g_wp_current_ax1.set_data([pl.global_trajectory.path_x[pl.global_trajectory.current_wp]], [pl.global_trajectory.path_y[pl.global_trajectory.current_wp]])
            g_wp_current_ax2.set_data([pl.global_trajectory.path_s[pl.global_trajectory.current_wp]], [pl.global_trajectory.path_d[pl.global_trajectory.current_wp]])
            g_wp_next_ax1.set_data([pl.global_trajectory.path_x[pl.global_trajectory.next_wp]], [pl.global_trajectory.path_y[pl.global_trajectory.next_wp]])
            g_wp_next_ax2.set_data([pl.global_trajectory.path_s[pl.global_trajectory.next_wp]], [pl.global_trajectory.path_d[pl.global_trajectory.next_wp]])

# TODO: currently shows only two levels only
def update_lattice_graph_plots(pl, show_plot = True):
    if not show_plot or len(pl.lattice_graph) == 0:
        # clear all lattice graph plots
        for line in lattice_graph_plots_ax1 + lattice_graph_endpoints_ax1 + lattice_graph_plots_ax2 + lattice_graph_endpoints_ax2:
            line.set_data([], [])
        return 

    edge_index = 0
    for k, v in pl.lattice_graph.items():
        if edge_index < max_lattice_edges:
            lattice_graph_plots_ax1[edge_index].set_data(v.local_trajectory.path_x, v.local_trajectory.path_y)
            lattice_graph_plots_ax2[edge_index].set_data(v.local_trajectory.path_s_with_respect_to_parent,
                                                          v.local_trajectory.path_d_with_respect_to_parent)
            lattice_graph_endpoints_ax1[edge_index].set_data([v.local_trajectory.path_x[-1]], [v.local_trajectory.path_y[-1]])
            lattice_graph_endpoints_ax2[edge_index].set_data([v.local_trajectory.path_s_with_respect_to_parent[-1]],
                                                             [v.local_trajectory.path_d_with_respect_to_parent[-1]])
            edge_index += 1
        for next_edge in v.next_edges:
            if edge_index < max_lattice_edges:
                lattice_graph_plots_ax1[edge_index].set_data(next_edge.local_trajectory.path_x, next_edge.local_trajectory.path_y)
                lattice_graph_plots_ax2[edge_index].set_data(next_edge.local_trajectory.path_s_with_respect_to_parent,
                                                              next_edge.local_trajectory.path_d_with_respect_to_parent)
                lattice_graph_endpoints_ax1[edge_index].set_data([next_edge.local_trajectory.path_x[-1]], [next_edge.local_trajectory.path_y[-1]])
                lattice_graph_endpoints_ax2[edge_index].set_data([next_edge.local_trajectory.path_s_with_respect_to_parent[-1]], [next_edge.local_trajectory.path_d_with_respect_to_parent[-1]])
                edge_index += 1

def update_local_plan_plots(pl, show_plot = True):
    if not show_plot or pl.selected_edge is None:
        selected_edge_plot_ax1.set_data([],[])
        selected_edge_plot_ax2.set_data([],[])
        selected_next_edge_plot_ax1.set_data([],[])
        selected_next_edge_plot_ax2.set_data([],[])
        current_wp_plot_ax1.set_data([], [])
        current_wp_plot_ax2.set_data([], [])
        next_wp_plot_ax1.set_data([], [])
        next_wp_plot_ax2.set_data([], [])

    elif pl.selected_edge is not None:
        x, y = pl.selected_edge.local_trajectory.get_current_xy()
        s = pl.selected_edge.local_trajectory.path_s_with_respect_to_parent[pl.selected_edge.local_trajectory.current_wp]
        d = pl.selected_edge.local_trajectory.path_d_with_respect_to_parent[pl.selected_edge.local_trajectory.current_wp]
    
        x_n, y_n = pl.selected_edge.local_trajectory.get_xy_by_waypoint(pl.selected_edge.local_trajectory.next_wp)
        s_n = pl.selected_edge.local_trajectory.path_s_with_respect_to_parent[pl.selected_edge.local_trajectory.next_wp]
        d_n = pl.selected_edge.local_trajectory.path_d_with_respect_to_parent[pl.selected_edge.local_trajectory.next_wp]
    
        current_wp_plot_ax1.set_data([x], [y])
        current_wp_plot_ax2.set_data([s], [d])
        
        next_wp_plot_ax1.set_data([x_n], [y_n])
        next_wp_plot_ax2.set_data([s_n], [d_n])
        
        v = pl.selected_edge
        selected_edge_plot_ax1.set_data(v.local_trajectory.path_x, v.local_trajectory.path_y)
        selected_edge_plot_ax2.set_data(v.local_trajectory.path_s_with_respect_to_parent, v.local_trajectory.path_d_with_respect_to_parent)
    
        if v.selected_next_edge is not None:
            selected_next_edge_plot_ax1.set_data(v.selected_next_edge.local_trajectory.path_x, v.selected_next_edge.local_trajectory.path_y)
            selected_next_edge_plot_ax2.set_data(v.selected_next_edge.local_trajectory.path_s_with_respect_to_parent,
                                                  v.selected_next_edge.local_trajectory.path_d_with_respect_to_parent)




def update_state_plots(state, show_plot = True):
    if not show_plot:
        car_heading_plot.set_data([], [])
        car_location_plot.set_data([], [])
        car_rect.set_xy((0, 0))
        car_rect.set_width(0)
        car_rect.set_height(0)
        return

    car_L_f = state.L_f
    car_L_r = state.length - car_L_f

    car_x_front = state.x + car_L_f * np.cos(state.theta)
    car_y_front = state.y + car_L_f * np.sin(state.theta)
    car_x_rear = state.x - car_L_r * np.cos(state.theta)
    car_y_rear = state.y - car_L_r * np.sin(state.theta)

    car_heading_plot.set_data([car_x_front, car_x_rear], [car_y_front, car_y_rear])
    car_location_plot.set_data([state.x], [state.y])

    car_rect.set_width(state.length)
    car_rect.set_height(state.width)

    # Calculate the center of the car
    car_center_x = state.x
    car_center_y = state.y

    # Calculate the four corners of the rectangle
    corners_x = np.array([car_center_x - state.length / 2, car_center_x + state.length / 2, car_center_x + state.length / 2, car_center_x - state.length / 2])
    corners_y = np.array([car_center_y - state.width / 2, car_center_y - state.width / 2, car_center_y + state.width / 2, car_center_y + state.width / 2])

    # Rotate the corners around the center of the car
    rotation_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)], [np.sin(state.theta), np.cos(state.theta)]])
    rotated_corners = np.dot(rotation_matrix, np.array([corners_x - car_center_x, corners_y - car_center_y]))
    rotated_corners_x = rotated_corners[0, :] + car_center_x
    rotated_corners_y = rotated_corners[1, :] + car_center_y

    # Update the rectangle position
    car_rect.set_xy((rotated_corners_x[0], rotated_corners_y[0]))
    car_rect.set_angle(np.degrees(state.theta))


if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()
