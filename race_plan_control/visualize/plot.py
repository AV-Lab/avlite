from plan.planner import Planner
from plan.lattice import Edge
from race_plan_control.execute.executer import Executer
from race_plan_control.perceive.state import VehicleState
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt

from race_plan_control.plan.trajectory import Trajectory


MAX_LATTICE_SIZE = 30
MAX_PLAN_LENGTH = 5

# plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1)


def initiate_plots(max_lattice_size=30, max_plan_length=5):
    global MAX_LATTICE_SIZE, MAX_PLAN_LENGTH
    MAX_LATTICE_SIZE = max_lattice_size
    MAX_PLAN_LENGTH = max_plan_length
    global lattice_graph_plots_ax1, lattice_graph_plots_ax2, lattice_graph_endpoints_ax1, lattice_graph_endpoints_ax2, local_plan_plots_ax1, local_plan_plots_ax2

    # Initialize lattice graph plots
    lattice_graph_plots_ax1 = []
    lattice_graph_plots_ax2 = []
    lattice_graph_endpoints_ax1 = []
    lattice_graph_endpoints_ax2 = []

    # initialize local plan
    local_plan_plots_ax1 = []
    local_plan_plots_ax2 = []

    # Assuming a maximum number of lattice graph edges
    for _ in range(MAX_LATTICE_SIZE):
        (line_ax1,) = ax1.plot([], [], "b--", color="lightskyblue", alpha=0.6)
        (line_ax2,) = ax2.plot([], [], "b--", color="lightskyblue", alpha=0.6)
        (endpoint_ax1,) = ax1.plot([], [], "bo", alpha=0.6)
        (endpoint_ax2,) = ax2.plot([], [], "bo", alpha=0.6)
        lattice_graph_plots_ax1.append(line_ax1)
        lattice_graph_plots_ax2.append(line_ax2)
        lattice_graph_endpoints_ax1.append(endpoint_ax1)
        lattice_graph_endpoints_ax2.append(endpoint_ax2)

    # local plan
    for i in range(MAX_PLAN_LENGTH):
        (local_plan_ax1,) = ax1.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
        (local_plan_ax2,) = ax2.plot([], [], "r-", label="Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
        local_plan_plots_ax1.append(local_plan_ax1)
        local_plan_plots_ax2.append(local_plan_ax2)

    return (
        lattice_graph_plots_ax1,
        lattice_graph_plots_ax2,
        lattice_graph_endpoints_ax1,
        lattice_graph_endpoints_ax2,
        local_plan_plots_ax1,
        local_plan_plots_ax2,
    )


(left_boundry_x1,) = ax1.plot(
    [], [], color="orange", label="Left Boundary", linewidth=2
)  # Change color and label as needed
(right_boundry_x1,) = ax1.plot(
    [], [], color="tan", label="Right Boundary", linewidth=2
)  # Change color and label as needed
left_boundry_ax2 = ax2.scatter([], [], color="orange", s=5, label="Left Boundary (Ref)")
right_boundry_ax2 = ax2.scatter([], [], color="tan", s=5, label="Right Boundary (Ref)")

(reference_trajectory_ax1,) = ax1.plot(
    [], [], "gray", label="Reference Trajectory", linewidth=2
)  # reference path in gray
reference_trajectory_ax2 = ax2.scatter(
    [], [], s=5, alpha=0.5, color="gray", label="Reference Trajectory"
)  # reference path in blue

(last_locs_ax1,) = ax1.plot([], [], "g-", label="Last 100 Locations", linewidth=2)  # Plot the last 100 points in green
(planner_loc_ax1,) = ax1.plot([], [], "ro", markersize=10, label="Planner Location")

(last_locs_ax2,) = ax2.plot([], [], "g-", label="Last 100 Locations", linewidth=2)  # Plot the last 100 points in green
(planner_loc_ax2,) = ax2.plot([], [], "ro", markersize=10, label="Planner Location")


(g_wp_current_ax1,) = ax1.plot([], [], "g", markersize=13, label="G WP: Curent", marker="o", fillstyle="none")
(g_wp_current_ax2,) = ax2.plot([], [], "g", markersize=13, label="G WP: Curent", marker="o", fillstyle="none")

(g_wp_next_ax1,) = ax1.plot([], [], "gx", markersize=13, label="G WP: Next")
(g_wp_next_ax2,) = ax2.plot([], [], "gx", markersize=13, label="G WP: Next")


# Local Plan initialization
# Initialize lattice graph plots
lattice_graph_plots_ax1 = []
lattice_graph_plots_ax2 = []
lattice_graph_endpoints_ax1 = []
lattice_graph_endpoints_ax2 = []

# initialize local plan
local_plan_plots_ax1 = []
local_plan_plots_ax2 = []

(current_wp_plot_ax1,) = ax1.plot([], [], "bo", markersize=15, label="L WP: Current", fillstyle="none")
(current_wp_plot_ax2,) = ax2.plot([], [], "bo", markersize=15, label="L WP: Current", fillstyle="none")
(next_wp_plot_ax1,) = ax1.plot([], [], "bx", markersize=15, label="L WP: Next", fillstyle="none")
(next_wp_plot_ax2,) = ax2.plot([], [], "bx", markersize=15, label="L WP: Next", fillstyle="none")

# State Init
(car_heading_plot,) = ax1.plot([], [], "k-", color="darkslategray", label="Car Heading")
(car_location_plot,) = ax1.plot([], [], "ko", color="darkslategray", markersize=7, label="Car Location")


ego_vehicle_ax1 = plt.Polygon(np.empty((0, 2)), closed=True, edgecolor="r", facecolor="azure", alpha=0.7)
ego_vehicle_ax2 = plt.Polygon(np.empty((0, 2)), closed=True, edgecolor="r", facecolor="azure", alpha=0.7)
ax1.add_patch(ego_vehicle_ax1)
ax2.add_patch(ego_vehicle_ax2)


# Assuming a maximum number of lattice graph edges
for _ in range(MAX_LATTICE_SIZE):
    (line_ax1,) = ax1.plot([], [], "b--", color="lightskyblue", alpha=0.6)
    (line_ax2,) = ax2.plot([], [], "b--", color="lightskyblue", alpha=0.6)
    (endpoint_ax1,) = ax1.plot([], [], "bo", alpha=0.6)
    (endpoint_ax2,) = ax2.plot([], [], "bo", alpha=0.6)
    lattice_graph_plots_ax1.append(line_ax1)
    lattice_graph_plots_ax2.append(line_ax2)
    lattice_graph_endpoints_ax1.append(endpoint_ax1)
    lattice_graph_endpoints_ax2.append(endpoint_ax2)

# local plan
for i in range(MAX_PLAN_LENGTH):
    (local_plan_ax1,) = ax1.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
    (local_plan_ax2,) = ax2.plot([], [], "r-", label="Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
    local_plan_plots_ax1.append(local_plan_ax1)
    local_plan_plots_ax2.append(local_plan_ax2)


ax2.set_title("Frenet Coordinate")
ax1.set_aspect("equal")
ax2.set_aspect("equal")


legend_ax = fig.add_axes([0.0, -0.013, 1, 0.1])
legend_ax.legend(*ax1.get_legend_handles_labels(), loc="center", ncol=7, borderaxespad=0.0, fontsize=7, framealpha=0.3)
legend_ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)


def plot(
    exec: Executer,
    aspect_ratio=4.0,
    frenet_zoom=15,
    xy_zoom=30,
    show_legend=True,
    plot_last_pts=True,
    plot_global_plan=True,
    plot_local_plan=True,
    plot_local_lattice=True,
    plot_state=True,
    num_plot_last_pts=100,
):

    legend_ax.set_visible(show_legend)

    # For Zoom in
    if xy_zoom is not None:
        ax1.set_xlim(
            exec.planner.traversed_x[-1] - xy_zoom,
            exec.planner.traversed_x[-1] + xy_zoom,
        )
        ax1.set_ylim(
            exec.planner.traversed_y[-1] - xy_zoom / aspect_ratio / 2,
            exec.planner.traversed_y[-1] + xy_zoom / aspect_ratio / 2,
        )
    if frenet_zoom is not None:
        ax2.set_xlim(
            exec.planner.traversed_s[-1] - frenet_zoom / 2,
            exec.planner.traversed_s[-1] + 1.5 * frenet_zoom,
        )
        ax2.set_ylim(-frenet_zoom / aspect_ratio / 2, frenet_zoom / aspect_ratio / 2)

    if plot_last_pts and num_plot_last_pts > 0:
        last_locs_ax1.set_data(
            exec.planner.traversed_x[-num_plot_last_pts:],
            exec.planner.traversed_y[-num_plot_last_pts:],
        )
        planner_loc_ax1.set_data([exec.planner.traversed_x[-1]], [exec.planner.traversed_y[-1]])

        last_locs_ax2.set_data(
            exec.planner.traversed_s[-num_plot_last_pts:],
            exec.planner.traversed_d[-num_plot_last_pts:],
        )
        planner_loc_ax2.set_data([exec.planner.traversed_s[-1]], [exec.planner.traversed_d[-1]])
    else:
        last_locs_ax1.set_data([], [])
        planner_loc_ax1.set_data([], [])
        last_locs_ax2.set_data([], [])
        planner_loc_ax2.set_data([], [])

    # update global Plan
    update_global_plan_plots(exec.planner, plot_global_plan)

    # update local plan
    update_lattice_graph_plots(exec.planner, plot_local_lattice)
    update_local_plan_plots(exec.planner, plot_local_plan)

    # update state
    update_state_plots(exec.ego_state, exec.planner.global_trajectory, plot_state)

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


def update_global_plan_plots(pl: Planner, show_plot=True):
    global initialized
    global toggle_plot
    if not initialized:
        # Plot track boundaries
        left_boundry_x1.set_data(pl.ref_left_boundary_x, pl.ref_left_boundary_y)
        right_boundry_x1.set_data(pl.ref_right_boundary_x, pl.ref_right_boundary_y)
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
        reference_trajectory_ax1.set_data([], [])
        reference_trajectory_ax2.set_offsets(np.c_[[], []])
        toggle_plot = True
        return
    elif initialized and toggle_plot:
        reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)
        reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.global_trajectory.path_d])
        toggle_plot = False

    if pl.global_trajectory.next_wp is not None:
        g_wp_current_ax1.set_data(
            [pl.global_trajectory.path_x[pl.global_trajectory.current_wp]],
            [pl.global_trajectory.path_y[pl.global_trajectory.current_wp]],
        )
        g_wp_current_ax2.set_data(
            [pl.global_trajectory.path_s[pl.global_trajectory.current_wp]],
            [pl.global_trajectory.path_d[pl.global_trajectory.current_wp]],
        )
        g_wp_next_ax1.set_data(
            [pl.global_trajectory.path_x[pl.global_trajectory.next_wp]],
            [pl.global_trajectory.path_y[pl.global_trajectory.next_wp]],
        )
        g_wp_next_ax2.set_data(
            [pl.global_trajectory.path_s[pl.global_trajectory.next_wp]],
            [pl.global_trajectory.path_d[pl.global_trajectory.next_wp]],
        )


edge_index = 0


def update_lattice_graph_plots(pl: Planner, show_plot=True):
    global edge_index
    if not show_plot or len(pl.next_edges) == 0:
        # clear all lattice graph plots
        for line in (
            lattice_graph_plots_ax1
            + lattice_graph_endpoints_ax1
            + lattice_graph_plots_ax2
            + lattice_graph_endpoints_ax2
        ):
            line.set_data([], [])
        return

    edge_index = 0
    __update_lattice_edge_plots(edge_index, pl=pl)


def __update_lattice_edge_plots(v: Edge = None, pl: Planner = None, level: int = 0):
    global edge_index
    edges = pl.next_edges if pl is not None else v.next_edges
    for next_edge in edges:
        if edge_index < MAX_LATTICE_SIZE:
            lattice_graph_plots_ax1[edge_index].set_data(
                next_edge.local_trajectory.path_x, next_edge.local_trajectory.path_y
            )
            lattice_graph_plots_ax2[edge_index].set_data(
                next_edge.local_trajectory.path_s_from_parent,
                next_edge.local_trajectory.path_d_from_parent,
            )
            lattice_graph_endpoints_ax1[edge_index].set_data(
                [next_edge.local_trajectory.path_x[-1]],
                [next_edge.local_trajectory.path_y[-1]],
            )
            lattice_graph_endpoints_ax2[edge_index].set_data(
                [next_edge.local_trajectory.path_s_from_parent[-1]],
                [next_edge.local_trajectory.path_d_from_parent[-1]],
            )
            edge_index += 1
            __update_lattice_edge_plots(v=next_edge, level=level + 1)


def update_local_plan_plots(pl: Planner, show_plot=True):
    if not show_plot or pl.selected_next_edge is None:
        for i in range(MAX_PLAN_LENGTH):
            local_plan_plots_ax1[i].set_data([], [])
            local_plan_plots_ax2[i].set_data([], [])
        current_wp_plot_ax1.set_data([], [])
        current_wp_plot_ax2.set_data([], [])
        next_wp_plot_ax1.set_data([], [])
        next_wp_plot_ax2.set_data([], [])

    elif pl.selected_next_edge is not None:
        x, y = pl.selected_next_edge.local_trajectory.get_current_xy()
        s = pl.selected_next_edge.local_trajectory.path_s_from_parent[pl.selected_next_edge.local_trajectory.current_wp]
        d = pl.selected_next_edge.local_trajectory.path_d_from_parent[pl.selected_next_edge.local_trajectory.current_wp]

        x_n, y_n = pl.selected_next_edge.local_trajectory.get_xy_by_waypoint(
            pl.selected_next_edge.local_trajectory.next_wp
        )
        s_n = pl.selected_next_edge.local_trajectory.path_s_from_parent[pl.selected_next_edge.local_trajectory.next_wp]
        d_n = pl.selected_next_edge.local_trajectory.path_d_from_parent[pl.selected_next_edge.local_trajectory.next_wp]

        # waypoints
        current_wp_plot_ax1.set_data([x], [y])
        current_wp_plot_ax2.set_data([s], [d])

        next_wp_plot_ax1.set_data([x_n], [y_n])
        next_wp_plot_ax2.set_data([s_n], [d_n])

        v = pl.selected_next_edge

        __update_local_plan_plots(v)


def __update_local_plan_plots(v: Edge, index: int = 0):
    if v is not None:
        local_plan_plots_ax1[index].set_data(
            v.local_trajectory.path_x,
            v.local_trajectory.path_y,
        )
        local_plan_plots_ax2[index].set_data(
            v.local_trajectory.path_s_from_parent,
            v.local_trajectory.path_d_from_parent,
        )
        __update_local_plan_plots(v.selected_next_edge, index + 1)


def update_state_plots(state: VehicleState, global_trajectory: Trajectory, show_plot=True):
    if not show_plot:
        car_heading_plot.set_data([], [])
        car_location_plot.set_data([], [])
        ego_vehicle_ax1.set_xy(np.empty((0, 2)))
        ego_vehicle_ax2.set_xy(np.empty((0, 2)))
        return

    car_L_f = state.L_f
    car_L_r = state.length - car_L_f

    car_x_front = state.x + car_L_f * np.cos(state.theta)
    car_y_front = state.y + car_L_f * np.sin(state.theta)
    car_x_rear = state.x - car_L_r * np.cos(state.theta)
    car_y_rear = state.y - car_L_r * np.sin(state.theta)

    car_heading_plot.set_data([car_x_front, car_x_rear], [car_y_front, car_y_rear])
    car_location_plot.set_data([state.x], [state.y])

    ego_vehicle_ax1.set_xy(state.get_corners())

    # sd_corners = [p for p in zip(* global_trajectory.convert_xy_path_to_sd_path(state.get_corners()))]
    sd_corners = global_trajectory.convert_xy_path_to_sd_path_np(state.get_corners())

    if np.abs(sd_corners[0][0] - sd_corners[1][0]) < 10:
        ego_vehicle_ax2.set_xy(np.array(sd_corners))
    else:
        ego_vehicle_ax2.set_xy(np.empty((0, 2)))


def set_plot_theme(bg_color="white", fg_color="black"):
    fig.patch.set_facecolor(bg_color)
    ax1.patch.set_facecolor(bg_color)
    ax2.patch.set_facecolor(bg_color)
    ax2.set_title("Frenet Coordinate", color=fg_color)
    # Set titles and labels to white
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)
        ax.tick_params(axis="both", colors=fg_color)  # Set tick colors to white
        ax.xaxis.label.set_color(fg_color)  # Set x-axis label color to white
        ax.yaxis.label.set_color(fg_color)  # Set y-axis label color to white

    redraw_plots()


if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
