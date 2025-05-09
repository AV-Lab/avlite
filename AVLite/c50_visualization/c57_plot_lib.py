from c10_perception.c12_perception_strategy import PerceptionModel
from c10_perception.c11_perception_model import EgoState
from c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from c20_planning.c27_lattice import Edge
from c20_planning.c28_trajectory import Trajectory
from c40_execution.c42_sync_executer import SyncExecuter
from c20_planning.c25_hdmap_global_planner import HDMapGlobalPlanner, HDMap


from typing import cast
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx

import logging

log = logging.getLogger(__name__)

class GlobalPlot(ABC):
    def __init__(self, figsize=(8, 10), name="Global Plot"):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.name = name
        self.ax.set_title(self.name)
        self.ax.grid(True)
        self.ax.set_aspect('equal') 
        
        self.background = None  # For blitting
        
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)
        self.start, = self.ax.plot([], [], 'bo', markersize=14, label="Start", zorder = 3)
        self.start_text = self.ax.text(1000, 1000, 'S', fontsize=12, color='white', zorder=4, ha='center', va='center')
        self.goal, = self.ax.plot([], [], 'go', markersize=14, label="Goal", zorder = 3)    
        self.goal_text = self.ax.text(1000, 1000 , 'G', fontsize=12, color='white', zorder=4, ha='center', va='center')
        self.vehicle_location, = self.ax.plot([], [], 'ro', markersize=14, label="Planner Location", zorder=3)
        self.vehicle_location_text = self.ax.text(0, 0, 'L', fontsize=12, color='white', zorder=4, ha='center', va='center')

        self.fig.legend(loc="upper right", fontsize=8, framealpha=0.3)
        
        self.map_min_x = None
        self.map_min_y = None
        self.map_max_x = None
        self.map_max_y = None
        self.map_plotted = False


    def plot(self,exec:SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True, follow_vehicle=True):
        if not self.map_plotted:
            self.plot_map(exec)

        start_x, start_y = exec.global_planner.global_plan.start_point
        goal_x, goal_y = exec.global_planner.global_plan.goal_point
        self.vehicle_x, self.vehicle_y = exec.ego_state.x, exec.ego_state.y
        self.vehicle_location.set_data([self.vehicle_x], [self.vehicle_y ])
        self.vehicle_location_text.set_position((self.vehicle_x, self.vehicle_y))

        
        self.adjust_zoom(zoom, aspect_ratio)

        if not show_legend:
            self.ax.get_legend().remove() if self.ax.get_legend() else None

        # self.ax.set_aspect(aspect_ratio)
        self.fig.canvas.draw()

    @abstractmethod
    def plot_map(self, exec:SyncExecuter):
        pass
    
    def adjust_zoom(self, zoom, aspect_ratio):
        """Adjust the zoom level and aspect ratio of the plot"""
        if self.map_min_x is not None and self.map_min_y is not None and self.map_max_x is not None and self.map_max_y is not None:
            # Set view limits
            mi_x = self.map_min_x - zoom
            mi_y = self.map_min_y - zoom/aspect_ratio
            ma_x = self.map_max_x + zoom
            ma_y = self.map_max_y + zoom/aspect_ratio
            if mi_x < ma_x and mi_y < ma_y:
                if self.map_min_x < self.vehicle_x < self.map_max_x and self.map_min_y < self.vehicle_y < self.map_max_y \
                     and zoom < self.map_max_x - self.map_min_x and zoom/aspect_ratio < self.map_max_y - self.map_min_y:

                    self.ax.set_xlim(self.vehicle_x - zoom, self.vehicle_x + zoom)
                    self.ax.set_ylim(self.vehicle_y - zoom/aspect_ratio, self.vehicle_y + zoom/aspect_ratio)
                else:
                    self.ax.set_xlim(self.map_min_x-20, self.map_max_x+20)
                    self.ax.set_ylim(self.map_min_y-20/aspect_ratio, self.map_max_y + 20/aspect_ratio)

    def set_start(self, x, y):
        """Set the start point"""
        self.start.set_data([x], [y])
        # self.ax.draw_artist(self.start)
        # self.fig.canvas.blit(self.ax.bbox)
        self.start_text.set_position((x, y))
        self.fig.canvas.draw()

    def set_goal(self, x, y):
        """Set the goal point"""
        self.goal.set_data([x], [y])
        self.goal_text.set_position((x, y))
        # self.ax.draw_artist(self.goal)
        # self.fig.canvas.blit(self.ax.bbox)
        
        self.fig.canvas.draw()


    def set_plot_theme(self, bg_color="white", fg_color="black"):
        """Set the plot theme colors"""
        # Apply background color with no transparency
        self.fig.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        
        # Set axis, ticks, and label colors
        for spine in self.ax.spines.values():
            spine.set_edgecolor(fg_color)
        
        self.ax.tick_params(axis="both", colors=fg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        
        # Set grid color with proper alpha for visibility
        self.ax.grid(False, color=fg_color, alpha=0.3)
        self.ax.set_title(label=self.name, color=fg_color)
        
        # Apply redraw
        self.fig.canvas.draw()
        
        log.debug(f"Global plot theme set to {bg_color} background and {fg_color} foreground.")


class GlobalRacePlot(GlobalPlot):
    def __init__(self, figsize=(8, 10)):
        super().__init__(figsize, name = "Global Race Plot")
        # Create plot elements with empty data - they'll be updated later
        self.left_boundary, = self.ax.plot([], [], 'orange', linewidth=3, label="Left Boundary")
        self.right_boundary, = self.ax.plot([], [], 'tan', linewidth=3, label="Right Boundary")
        self.reference_trajectory, = self.ax.plot([], [], 'gray', linewidth=3, label="Global Trajectory")
        
      
        self.ax.legend()
        
        # Adjust layout to align with LocalPlot
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        super().set_plot_theme(bg_color, fg_color)

        # Use the same colors as LocalPlot, not black/white specific colors
        self.left_boundary.set_color("orange")
        self.right_boundary.set_color("tan")
        self.reference_trajectory.set_color("gray")
        self.vehicle_location.set_color("red")
        
        
    def plot_map(self, exec: SyncExecuter):
        """Update the plot with current data"""

        log.debug("Plotting Race Global Plot")
        self.left_boundary.set_data(exec.local_planner.ref_left_boundary_x, exec.local_planner.ref_left_boundary_y)
        self.right_boundary.set_data(exec.local_planner.ref_right_boundary_x, exec.local_planner.ref_right_boundary_y)
        self.reference_trajectory.set_data(exec.local_planner.global_trajectory.path_x, exec.local_planner.global_trajectory.path_y)
        
        self.map_min_x = min(exec.local_planner.ref_left_boundary_x)
        self.map_max_x = max(exec.local_planner.ref_right_boundary_x)
        self.map_min_y = min(exec.local_planner.ref_left_boundary_y)
        self.map_max_y = max(exec.local_planner.ref_right_boundary_y)
        self.map_plotted = True
            




class GlobalHDMapPlot(GlobalPlot):
    def __init__(self, figsize=(10, 10), MAX_ROAD_PATH=20):
        super().__init__(figsize, name="HD Map Road Network")

        self.closest_lane, *_ = self.ax.plot([], [], 'ro', markersize=5, label="Closest Lane", zorder=3)
        self.closest_road, *_ = self.ax.plot([], [], 'ro', markersize=5, alpha=.1,  label="Closest Road", zorder=3)

        self.road_path_plots = []
        for i in range(MAX_ROAD_PATH):
           pl , *_ = self.ax.plot([], [], 'r-', linewidth=2, alpha=0.5, label="Road Path", zorder=3)
           self.road_path_plots.append(pl)
        

    def show_closest_road_lane(self,  x:int, y:int, map:HDMap):
        """Show the closest road and lane to the given coordinates"""
        r = map.find_nearest_road(x,y)
        # l = map.point_to_lane.get((x, y))

        if r is not None:
            self.closest_road.set_data(r.center_line[0], r.center_line[1])
            self.fig.canvas.draw()

    def plot_road_path(self, road_path: list[HDMap.Road]):
        """Plot the road path"""
        log.debug("Plotting Road Path: length = %d", len(road_path))
        self.__clear_road_path_plots()
        for i,r in enumerate(road_path):
            self.road_path_plots[i].set_data(r.center_line[0], r.center_line[1])
                
        self.fig.canvas.draw()

    def __clear_road_path_plots(self):
        """Clear the road path plots"""
        for i in range(len(self.road_path_plots)):
            self.road_path_plots[i].set_data([], [])
        
    def plot_map(self, exec:SyncExecuter):
        """Implement the abstract method from GlobalPlot"""
        
        if not hasattr(exec.global_planner, "hdmap"):
            log.error("HDMap not found in the global planner.")
            return

        global_planner = exec.global_planner
        global_planner = cast(HDMapGlobalPlanner,global_planner)
        hdmap = global_planner.hdmap


        all_x_coords = []
        all_y_coords = []

        # for r in hdmap.roads:
            # self.ax.plot(r.center_line[0], r.center_line[1], color='red', linewidth=2, alpha=0.5)

        for l in hdmap.lanes:
            if l.type == "driving":
                color = "green"
                alpha = 0.7
            elif l.type == "shoulder":
                color = "orange"
                alpha = 0.5
            else:
                color = "gray"
                alpha = 0.3
            
            self.ax.plot(l.center_line[0], l.center_line[1], color=color, linewidth=2, alpha=alpha)
            all_x_coords.extend(l.center_line[0])
            all_y_coords.extend(l.center_line[1])

        self.map_min_x = min(all_x_coords)  
        self.map_min_y = min(all_y_coords)
        self.map_max_x = max(all_x_coords)
        self.map_max_y = max(all_y_coords)
        self.map_plotted = True
            

class LocalPlot:
    def __init__(self, max_lattice_size=21, max_plan_length=5, max_agent_count=12):
        self.MAX_LATTICE_SIZE = max_lattice_size
        self.MAX_PLAN_LENGTH = max_plan_length
        self.MAX_AGENT_COUNT = max_agent_count

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.ax2.set_title("Frenet Coordinate", pad=-100)

        self.ax1.set_aspect("equal")
        self.ax2.set_aspect("equal")
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)
        # self.ax1.set_position([0, 0.5, 0.99, .5])  # [left, bottom, width, height]
        # self.ax2.set_position([0, 0.0, 0.99, .5])  # [left, bottom, width, height]

        self.lattice_graph_plots_ax1 = []
        self.lattice_graph_plots_ax2 = []
        self.lattice_graph_endpoints_ax1 = []
        self.lattice_graph_endpoints_ax2 = []
        self.local_plan_plots_ax1 = []
        self.local_plan_plots_ax2 = []

        for _ in range(self.MAX_LATTICE_SIZE):
            (line_ax1,) = self.ax1.plot([], [], "b--", color="lightskyblue", alpha=0.6)
            (line_ax2,) = self.ax2.plot([], [], "b--", color="lightskyblue", alpha=0.6)
            (endpoint_ax1,) = self.ax1.plot([], [], "bo", alpha=0.6)
            (endpoint_ax2,) = self.ax2.plot([], [], "bo", alpha=0.6)
            self.lattice_graph_plots_ax1.append(line_ax1)
            self.lattice_graph_plots_ax2.append(line_ax2)
            self.lattice_graph_endpoints_ax1.append(endpoint_ax1)
            self.lattice_graph_endpoints_ax2.append(endpoint_ax2)

        for i in range(self.MAX_PLAN_LENGTH):
            (local_plan_ax1,) = self.ax1.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
            (local_plan_ax2,) = self.ax2.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
            self.local_plan_plots_ax1.append(local_plan_ax1)
            self.local_plan_plots_ax2.append(local_plan_ax2)

        (self.left_boundry_x1,) = self.ax1.plot([], [], color="orange", label="Left Boundary", linewidth=2)
        (self.right_boundry_x1,) = self.ax1.plot([], [], color="tan", label="Right Boundary", linewidth=2)
        self.left_boundry_ax2 = self.ax2.scatter([], [], color="orange", s=5, label="Left Boundary (Ref)")
        self.right_boundry_ax2 = self.ax2.scatter([], [], color="tan", s=5, label="Right Boundary (Ref)")

        (self.reference_trajectory_ax1,) = self.ax1.plot([], [], "gray", label="Reference Trajectory", linewidth=2)
        self.reference_trajectory_ax2 = self.ax2.scatter(
            [], [], s=5, alpha=0.5, color="gray", label="Global Trajectory"
        )

        (self.last_locs_ax1,) = self.ax1.plot([], [], "g-", label="Last 100 Locations", linewidth=2)
        (self.planner_loc_ax1,) = self.ax1.plot([], [], "ro", markersize=10, label="Planner Location")

        (self.last_locs_ax2,) = self.ax2.plot([], [], "g-", label="Last 100 Locations", linewidth=2)
        (self.planner_loc_ax2,) = self.ax2.plot([], [], "ro", markersize=10, label="Planner Location")

        (self.g_wp_current_ax1,) = self.ax1.plot(
            [], [], "g", markersize=13, label="G WP: Curent", marker="o", fillstyle="none"
        )
        (self.g_wp_current_ax2,) = self.ax2.plot(
            [], [], "g", markersize=13, label="G WP: Curent", marker="o", fillstyle="none"
        )

        (self.g_wp_next_ax1,) = self.ax1.plot([], [], "gx", markersize=13, label="G WP: Next")
        (self.g_wp_next_ax2,) = self.ax2.plot([], [], "gx", markersize=13, label="G WP: Next")

        (self.current_wp_plot_ax1,) = self.ax1.plot(
            [], [], "bo", markersize=15, label="L WP: Current", fillstyle="none"
        )
        (self.current_wp_plot_ax2,) = self.ax2.plot(
            [], [], "bo", markersize=15, label="L WP: Current", fillstyle="none"
        )
        (self.next_wp_plot_ax1,) = self.ax1.plot([], [], "bx", markersize=15, label="L WP: Next", fillstyle="none")
        (self.next_wp_plot_ax2,) = self.ax2.plot([], [], "bx", markersize=15, label="L WP: Next", fillstyle="none")

        (self.car_heading_plot,) = self.ax1.plot([], [], "k-", color="darkslategray", label="Car Heading")
        (self.car_location_plot,) = self.ax1.plot(
            [], [], "ko", markersize=7, label="Car Location"
        )

        self.ego_vehicle_ax1 = Polygon(np.empty((0, 2)), closed=True, edgecolor="r", facecolor="azure", alpha=0.7)
        self.ego_vehicle_ax2 = Polygon(np.empty((0, 2)), closed=True, edgecolor="r", facecolor="azure", alpha=0.7)
        self.ax1.add_patch(self.ego_vehicle_ax1)
        self.ax2.add_patch(self.ego_vehicle_ax2)

        self.pm_plots_ax1 = []
        self.pm_plots_ax2 = []
        for _ in range(self.MAX_AGENT_COUNT):
            agent_vehicle_ax1 = Polygon(
                np.empty((0, 2)), closed=True, edgecolor="darkblue", facecolor="azure", alpha=0.6
            )
            agent_vehicle_ax2 = Polygon(
                np.empty((0, 2)), closed=True, edgecolor="darkblue", facecolor="azure", alpha=0.6
            )
            self.ax1.add_patch(agent_vehicle_ax1)
            self.ax2.add_patch(agent_vehicle_ax2)
            self.pm_plots_ax1.append(agent_vehicle_ax1)
            self.pm_plots_ax2.append(agent_vehicle_ax2)

        self.legend_ax = self.fig.add_axes([0.0, -0.013, 1, 0.1])
        self.legend_ax.legend(
            *self.ax1.get_legend_handles_labels(), loc="center", ncol=7, borderaxespad=0.0, fontsize=7, framealpha=0.3
        )
        self.legend_ax.axis("off")

    def plot(
        self,
        exec: SyncExecuter,
        aspect_ratio=4.0,
        frenet_zoom=15,
        xy_zoom=30,
        show_legend=True,
        plot_last_pts=True,
        plot_global_plan=True,
        plot_local_plan=True,
        plot_local_lattice=True,
        plot_state=True,
        plot_perception_model=True,
        num_plot_last_pts=100,
        global_follow_planner = False,
        frenet_follow_planner = False
    ):
        self.legend_ax.set_visible(show_legend)
        
        center_xy = exec.local_planner.location_xy if global_follow_planner else  (exec.ego_state.x, exec.ego_state.y)
        center_sd = exec.local_planner.location_sd if frenet_follow_planner else exec.local_planner.global_trajectory.convert_xy_to_sd(*center_xy)
        if xy_zoom is not None:
            self.ax1.set_xlim(center_xy[0] - xy_zoom, center_xy[0] + xy_zoom)
            self.ax1.set_ylim(
                center_xy[1] - xy_zoom / aspect_ratio / 2,
                center_xy[1] + xy_zoom / aspect_ratio / 2,
            )
        if frenet_zoom is not None:
            self.ax2.set_xlim(
                center_sd[0] - frenet_zoom / 2, center_sd[0] + 1.5 * frenet_zoom
            )
            self.ax2.set_ylim(-frenet_zoom / aspect_ratio / 2, frenet_zoom / aspect_ratio / 2)

        if plot_last_pts and num_plot_last_pts > 0:
            self.last_locs_ax1.set_data(
                exec.local_planner.traversed_x[-num_plot_last_pts:], exec.local_planner.traversed_y[-num_plot_last_pts:]
            )
            self.planner_loc_ax1.set_data([exec.local_planner.location_xy[0]], [exec.local_planner.location_xy[1]])

            self.last_locs_ax2.set_data(
                exec.local_planner.traversed_s[-num_plot_last_pts:], exec.local_planner.traversed_d[-num_plot_last_pts:]
            )
            self.planner_loc_ax2.set_data([exec.local_planner.location_sd[0]], [exec.local_planner.location_sd[1]])
        else:
            self.last_locs_ax1.set_data([], [])
            self.planner_loc_ax1.set_data([], [])
            self.last_locs_ax2.set_data([], [])
            self.planner_loc_ax2.set_data([], [])

        self.update_global_plan_plots(exec.local_planner, plot_global_plan)
        self.update_lattice_graph_plots(exec.local_planner, plot_local_lattice)
        self.update_local_plan_plots(exec.local_planner, plot_local_plan)
        self.update_state_plots(exec.ego_state, exec.local_planner.global_trajectory, plot_state)
        self.update_perception_model_plots(exec.pm, exec.local_planner.global_trajectory, plot_perception_model)
        self.redraw_plots()

    def redraw_plots(self):
        self.ax1.draw_artist(self.ax1.patch)
        self.ax2.draw_artist(self.ax2.patch)

        for line in self.ax1.lines:
            self.ax1.draw_artist(line)
        for line in self.ax2.lines:
            self.ax2.draw_artist(line)

        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)

    def update_global_plan_plots(self, pl: LocalPlannerStrategy, show_plot=True):
        if not hasattr(self, "initialized"):
            self.initialized = False
        if not hasattr(self, "toggle_plot"):
            self.toggle_plot = False

        if not self.initialized:
            self.left_boundry_x1.set_data(pl.ref_left_boundary_x, pl.ref_left_boundary_y)
            self.right_boundry_x1.set_data(pl.ref_right_boundary_x, pl.ref_right_boundary_y)
            self.left_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.ref_left_boundary_d])
            self.right_boundry_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.ref_right_boundary_d])
            self.reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)
            self.reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.global_trajectory.path_d])
            self.initialized = True

        if not show_plot:
            self.g_wp_current_ax1.set_data([], [])
            self.g_wp_current_ax2.set_data([], [])
            self.g_wp_next_ax1.set_data([], [])
            self.g_wp_next_ax2.set_data([], [])
            self.reference_trajectory_ax1.set_data([], [])
            self.reference_trajectory_ax2.set_offsets(np.c_[[], []])
            self.toggle_plot = True
            return
        elif self.initialized and self.toggle_plot:
            self.reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)
            self.reference_trajectory_ax2.set_offsets(np.c_[pl.global_trajectory.path_s, pl.global_trajectory.path_d])
            self.toggle_plot = False

        if pl.global_trajectory.next_wp is not None:
            self.g_wp_current_ax1.set_data(
                [pl.global_trajectory.path_x[pl.global_trajectory.current_wp]],
                [pl.global_trajectory.path_y[pl.global_trajectory.current_wp]],
            )
            self.g_wp_current_ax2.set_data(
                [pl.global_trajectory.path_s[pl.global_trajectory.current_wp]],
                [pl.global_trajectory.path_d[pl.global_trajectory.current_wp]],
            )
            self.g_wp_next_ax1.set_data(
                [pl.global_trajectory.path_x[pl.global_trajectory.next_wp]],
                [pl.global_trajectory.path_y[pl.global_trajectory.next_wp]],
            )
            self.g_wp_next_ax2.set_data(
                [pl.global_trajectory.path_s[pl.global_trajectory.next_wp]],
                [pl.global_trajectory.path_d[pl.global_trajectory.next_wp]],
            )

    def update_lattice_graph_plots(self, pl: LocalPlannerStrategy, show_plot=True):
        if not show_plot or len(pl.lattice.edges) == 0:
            for line in (
                self.lattice_graph_plots_ax1
                + self.lattice_graph_endpoints_ax1
                + self.lattice_graph_plots_ax2
                + self.lattice_graph_endpoints_ax2
            ):
                line.set_data([], [])
            return

        edge_index = 0
        for edge in pl.lattice.edges:
            if edge_index < self.MAX_LATTICE_SIZE:
                self.lattice_graph_plots_ax1[edge_index].set_data(
                    edge.local_trajectory.path_x, edge.local_trajectory.path_y
                )
                self.lattice_graph_plots_ax2[edge_index].set_data(
                    edge.local_trajectory.path_s_from_parent, edge.local_trajectory.path_d_from_parent
                )
                if edge.collision:
                    self.lattice_graph_plots_ax1[edge_index].set_color("firebrick")
                    self.lattice_graph_plots_ax2[edge_index].set_color("firebrick")
                else:
                    self.lattice_graph_plots_ax1[edge_index].set_color("lightskyblue")
                    self.lattice_graph_plots_ax2[edge_index].set_color("lightskyblue")

                self.lattice_graph_endpoints_ax1[edge_index].set_data(
                    [edge.local_trajectory.path_x[-1]], [edge.local_trajectory.path_y[-1]]
                )
                self.lattice_graph_endpoints_ax2[edge_index].set_data(
                    [edge.local_trajectory.path_s_from_parent[-1]], [edge.local_trajectory.path_d_from_parent[-1]]
                )
                edge_index += 1
            else:
                log.warning(
                    f"Lattice graph size exceeded: attempting to plot edge {edge_index+1} out of {self.MAX_LATTICE_SIZE}"
                )

    def update_local_plan_plots(self, pl: LocalPlannerStrategy, show_plot=True):
        if not show_plot or pl.selected_local_plan is None:
            self.__clear_local_plan_plots()
            self.current_wp_plot_ax1.set_data([], [])
            self.current_wp_plot_ax2.set_data([], [])
            self.next_wp_plot_ax1.set_data([], [])
            self.next_wp_plot_ax2.set_data([], [])
        elif pl.selected_local_plan is not None:
            x, y = pl.selected_local_plan.local_trajectory.get_current_xy()
            local_tj = pl.selected_local_plan.local_trajectory
            s = local_tj.path_s_from_parent[local_tj.current_wp]
            d = local_tj.path_d_from_parent[local_tj.current_wp]

            x_n, y_n = local_tj.get_xy_by_waypoint(local_tj.next_wp)
            s_n = local_tj.path_s_from_parent[local_tj.next_wp]
            d_n = local_tj.path_d_from_parent[local_tj.next_wp]

            self.current_wp_plot_ax1.set_data([x], [y])
            self.current_wp_plot_ax2.set_data([s], [d])

            self.next_wp_plot_ax1.set_data([x_n], [y_n])
            self.next_wp_plot_ax2.set_data([s_n], [d_n])

            v = pl.selected_local_plan
            self.__update_local_plan_plots(v, index=0, horizon=pl.planning_horizon)

    def __update_local_plan_plots(self, v: Edge, index: int = 0, horizon: int = None):
        if horizon is None:
            horizon = self.MAX_PLAN_LENGTH - 1
        if v is not None:
            self.local_plan_plots_ax1[index].set_data(v.local_trajectory.path_x, v.local_trajectory.path_y)
            self.local_plan_plots_ax2[index].set_data(
                v.local_trajectory.path_s_from_parent, v.local_trajectory.path_d_from_parent
            )
            self.__update_local_plan_plots(v.selected_next_local_plan, index + 1, horizon)
        elif index < horizon - 1:
            # log.info(f"Index: {index} is less than {self.MAX_PLAN_LENGTH - 1}")
            self.__clear_local_plan_plots(index=index)

    def __clear_local_plan_plots(self, index=0):
        for i in range(index, self.MAX_PLAN_LENGTH):
            self.local_plan_plots_ax1[i].set_data([], [])
            self.local_plan_plots_ax2[i].set_data([], [])

    def update_state_plots(self, state: EgoState, global_trajectory: Trajectory, show_plot=True):
        if not show_plot:
            self.car_heading_plot.set_data([], [])
            self.car_location_plot.set_data([], [])
            self.ego_vehicle_ax1.set_xy(np.empty((0, 2)))
            self.ego_vehicle_ax2.set_xy(np.empty((0, 2)))
            return


        car_L_f = state.L_f
        car_L_r = state.length - car_L_f

        car_x_front = state.x + car_L_f * np.cos(state.theta)
        car_y_front = state.y + car_L_f * np.sin(state.theta)
        car_x_rear = state.x - car_L_r * np.cos(state.theta)
        car_y_rear = state.y - car_L_r * np.sin(state.theta)

        self.car_heading_plot.set_data([car_x_front, car_x_rear], [car_y_front, car_y_rear])
        self.car_location_plot.set_data([state.x], [state.y])

        self.ego_vehicle_ax1.set_xy(state.get_bb_corners())
        sd_corners = global_trajectory.convert_xy_path_to_sd_path_np(state.get_bb_corners())

        if np.abs(sd_corners[0][0] - sd_corners[1][0]) < 10:
            self.ego_vehicle_ax2.set_xy(np.array(sd_corners))
        else:
            self.ego_vehicle_ax2.set_xy(np.empty((0, 2)))

    def update_perception_model_plots(self, pm: PerceptionModel, global_trajectory: Trajectory, show_plot=True):
        if not show_plot or len(pm.agent_vehicles) == 0:
            for i in range(self.MAX_AGENT_COUNT):
                self.pm_plots_ax1[i].set_xy(np.empty((0, 2)))
                self.pm_plots_ax2[i].set_xy(np.empty((0, 2)))
            return

        def transform(row):
            return global_trajectory.convert_xy_to_sd(row[0], row[1])

        for i, agent in enumerate(pm.agent_vehicles):
            if i >= self.MAX_AGENT_COUNT:
                log.warning(f"Exceeded maximum number of agents: {self.MAX_AGENT_COUNT}")
                break
            self.pm_plots_ax1[i].set_xy(agent.get_bb_corners())
            self.pm_plots_ax2[i].set_xy(agent.get_transformed_bb_corners(transform))

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        self.fig.patch.set_facecolor(bg_color)
        self.ax1.patch.set_facecolor(bg_color)
        self.ax2.patch.set_facecolor(bg_color)
        self.ax2.set_title("Frenet Coordinate", color=fg_color)
        # Set titles and labels to white
        for ax in [self.ax1, self.ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor(fg_color)
            ax.tick_params(axis="both", colors=fg_color)  # Set tick colors to white
            ax.xaxis.label.set_color(fg_color)  # Set x-axis label color to white
            ax.yaxis.label.set_color(fg_color)  # Set y-axis label color to white

        log.debug(f"Plot theme set to {bg_color} background and {fg_color} foreground.")
        self.redraw_plots()
