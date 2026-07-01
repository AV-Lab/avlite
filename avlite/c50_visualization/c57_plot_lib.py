from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c10_perception.c11_perception_model import EgoState, PredictionMode
from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c18_hdmap_parser import HDMap
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c28_lattice import Edge
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings

from typing import cast, Optional
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

import logging

log = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class GlobalPlot(ABC):
    def __init__(self, figsize=(8, 10), name="Global Plot"):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.name = name
        self.ax.grid(True)
        self.ax.set_aspect('equal') 
        
        self.background = None  # For blitting
        
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)
        self.start, = self.ax.plot([], [], 'bo', markersize=14, label="Start", zorder=7)
        self.start_text = self.ax.text(1000, 1000, 'S', fontsize=12, color='white', zorder=9, ha='center', va='center')
        self.goal, = self.ax.plot([], [], 'go', markersize=14, label="Goal", zorder=7)
        self.goal_text = self.ax.text(1000, 1000 , 'G', fontsize=12, color='white', zorder=9, ha='center', va='center')
        self.vehicle_location, = self.ax.plot([], [], 'ro', markersize=14, label="Planner Location", zorder=8)
        self.vehicle_location_text = self.ax.text(0, 0, 'L', fontsize=12, color='white', zorder=9, ha='center', va='center')

        self.orientation_arrow = None  # For the vehicle orientation arrow
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)

        # self.fig.legend(loc="upper right", fontsize=8, framealpha=0.3)
        
        self.map_min_x = None
        self.map_min_y = None
        self.map_max_x = None
        self.map_max_y = None
        self.view_width = None
        self.view_height = None
        self.map_plotted = False
        self._velocity_scale = "relative"


    def plot(self, exec:SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True, follow_vehicle=True, show_plan_boundaries=True, velocity_scale="relative", delta:Optional[tuple[float,float]]=None):
        self._velocity_scale = velocity_scale
        if not self.map_plotted:
            self.plot_map(exec.global_planner)

        self.plot_vehicle(exec.ego_state) 
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)

        # if not show_legend:
            # self.ax.get_legend().remove() if self.ax.get_legend() else None

        # self.ax.set_aspect(aspect_ratio)
        self.fig.canvas.draw()

    def plot_vehicle(self, ego:EgoState):
        """Plot the vehicle location"""
        self.vehicle_x, self.vehicle_y = ego.x, ego.y
        self.vehicle_location.set_data([self.vehicle_x], [self.vehicle_y ])
        self.vehicle_location_text.set_position((self.vehicle_x, self.vehicle_y))

    def clear_tmp_plots(self):
        """Clears temporary plots (e.g., cursor highlights)"""
        pass

    @abstractmethod
    def plot_map(self, exec:SyncExecuter):
        pass
    
    def adjust_center_and_zoom(self, zoom, aspect_ratio,center:Optional[tuple[float,float]]=None, delta:Optional[tuple[float,float]]=None):
        """Adjust the zoom level and aspect ratio of the plot"""
        if self.map_min_x is not None and self.map_min_y is not None and self.map_max_x is not None and self.map_max_y is not None:
            # Set view limits
            mi_x = self.map_min_x - zoom
            mi_y = self.map_min_y - zoom/aspect_ratio
            ma_x = self.map_max_x + zoom
            ma_y = self.map_max_y + zoom/aspect_ratio
            
            center_x, center_y = center if center else (self.vehicle_x, self.vehicle_y)
            if delta:
                center_x += delta[0]
                center_y += delta[1]


            if mi_x < ma_x and mi_y < ma_y:
                pad = 100
                if self.map_min_x - pad < center_x < self.map_max_x + pad and self.map_min_y - pad < center_y < self.map_max_y + pad \
                     and zoom < self.map_max_x - self.map_min_x and zoom/aspect_ratio < self.map_max_y - self.map_min_y:

                    self.ax.set_xlim(center_x - zoom, center_x + zoom)
                    self.ax.set_ylim(center_y - zoom/aspect_ratio, center_y + zoom/aspect_ratio)
                    self.view_width = zoom * 2
                    self.view_height = zoom / aspect_ratio * 2
                else:
                    map_width = self.map_max_x - self.map_min_x
                    map_height = self.map_max_y - self.map_min_y
                    
                    target_width = map_height * aspect_ratio  
                    target_height = map_width / aspect_ratio
                    
                    if target_width > map_width:
                        # Need to expand width
                        x_pad = (target_width - map_width) / 2
                        y_pad = 0
                    else:
                        # Need to expand height  
                        x_pad = 0
                        y_pad = (target_height - map_height) / 2
                    
                    self.ax.set_xlim(self.map_min_x - x_pad, self.map_max_x + x_pad)
                    self.ax.set_ylim(self.map_min_y - y_pad, self.map_max_y + y_pad)
                    self.view_width = map_width + x_pad * 2
                    self.view_height = map_height + y_pad * 2


    def set_start(self, x, y):
        """Set the start point"""
        self.start.set_data([x], [y])
        # self.ax.draw_artist(self.start)
        # self.fig.canvas.blit(self.ax.bbox)
        self.start_text.set_position((x, y))
        self.start_text.set_text("S")
        self.fig.canvas.draw()

    def set_goal(self, x, y):
        """Set the goal point"""
        self.goal.set_data([x], [y])
        self.goal_text.set_position((x, y))
        self.goal_text.set_text("G")
        # self.ax.draw_artist(self.goal)
        # self.fig.canvas.blit(self.ax.bbox)
        
        self.fig.canvas.draw()

    def show_vehicle_orientation(self, x, y, theta):
        """Show the vehicle orientation on the plot"""
        log.debug(f"Showing vehicle orientation at ({x}, {y}) with theta={theta}")

        if self.view_width is not None:
            length = min(self.view_width, self.view_height) * .15
        else:
            length = np.abs(self.map_max_x - self.map_min_x) * .2

        x2 = x + length * np.cos(theta)
        y2 = y + length * np.sin(theta)
        if self.orientation_arrow:
            self.orientation_arrow.remove()
        self.orientation_arrow = self.ax.annotate('', xy=(x2, y2), xytext=(x,y), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color="red", lw=5), zorder=9)
        
    def clear_tmp_plots(self):
        self.orientation_arrow.remove() if self.orientation_arrow else None
        self.orientation_arrow = None


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
        self.ax.grid(False)
        # self.ax.set_title(label=self.name, color=fg_color, y=-0.05)
        
        # Apply redraw
        self.fig.canvas.draw()
        
        log.debug(f"Global plot theme set to {bg_color} background and {fg_color} foreground.")
        

    def reset(self):
        self.map_plotted = False


class GlobalRacePlot(GlobalPlot):
    def __init__(self, figsize=(8, 10)):
        super().__init__(figsize, name = "Global Race Plot")
        # Create plot elements with empty data - they'll be updated later
        self.left_boundary, = self.ax.plot([], [], 'orange', linewidth=3, label="Left Boundary")
        self.right_boundary, = self.ax.plot([], [], 'tan', linewidth=3, label="Right Boundary")
        self.reference_trajectory = LineCollection([], cmap=_VELOCITY_CMAP, linewidths=5, zorder=4)
        self.ax.add_collection(self.reference_trajectory)
        
      
        # self.ax.legend()
        
        # Adjust layout to align with LocalPlot
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)

    def plot(self, exec: SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True,
             follow_vehicle=True, show_plan_boundaries: bool = True,
             velocity_scale: str = "relative",
             delta: Optional[tuple[float, float]] = None):
        self._velocity_scale = velocity_scale
        self.plot_map(exec.global_planner, show_plan_boundaries=show_plan_boundaries)
        self.plot_vehicle(exec.ego_state)
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)
        self.fig.canvas.draw()

    def clear_goal(self):
        """Clear the goal plot (no-op for race plot)."""
        self.goal.set_data([], [])
        self.goal_text.set_text("")
        self.fig.canvas.draw()

    def clear_road_path_plots(self):
        """Clear road path plots (no-op for race plot)."""
        pass

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        super().set_plot_theme(bg_color, fg_color)

        # Use the same colors as LocalPlot, not black/white specific colors
        self.left_boundary.set_color("orange")
        self.right_boundary.set_color("tan")
        self.vehicle_location.set_color("red")
        
        
    def plot_map(self, global_planner: GlobalPlannerStrategy, show_plan_boundaries: bool = True):
        """Update the plot with current data"""

        log.debug("Plotting Race Global Plot")
        plan = global_planner.global_plan
        if show_plan_boundaries:
            self.left_boundary.set_data(plan.left_boundary_x, plan.left_boundary_y)
            self.right_boundary.set_data(plan.right_boundary_x, plan.right_boundary_y)
        else:
            self.left_boundary.set_data([], [])
            self.right_boundary.set_data([], [])
        traj = plan.trajectory
        _update_velocity_colored_line(
            self.reference_trajectory, traj.path_x, traj.path_y, traj.velocity,
            velocity_scale=self._velocity_scale,
        )
        
        self.map_min_x = min(plan.left_boundary_x)
        self.map_max_x = max(plan.right_boundary_x)
        self.map_min_y = min(plan.left_boundary_y)
        self.map_max_y = max(plan.right_boundary_y)
        self.map_plotted = True
            


class GlobalHDMapPlot(GlobalPlot):
    def __init__(self, figsize=(10, 10), MAX_ROAD_PATH=20, MAX_SUCCS=10, MAX_PREDS=10):
        super().__init__(figsize, name="HD Map Road Network")
        
        # Gruvbox colors
        orange = "#d65d0e"
        light_orange = "#fe8019"
        light_aqua = "#8ec07c"
        aqua = "#689d6a"
        blue = "#076678"
        yellow = "yellow"
        red = 'red'
        purple='#b16286'
        txt_bg = '#1d2021'

        self.yellow = yellow; self.red = red; self.blue = blue

        self.closest_lane, *_ = self.ax.plot([], [], 'o-', color=yellow,  alpha=.2, label="Closest Lane", zorder=3)
        self.closest_road, *_ = self.ax.plot([], [], 'o-', color=red, alpha=.1,  label="Closest Road", zorder=3)
        self.closest_road_preds = []
        self.closest_road_succs = []
        for i in range(MAX_PREDS):
            p, *_ = self.ax.plot([], [], 'o-', color=orange,  alpha=.1, label="Closest Road Pred", zorder=3)
            self.closest_road_preds.append(p)
        for i in range(MAX_SUCCS):
            s, *_ = self.ax.plot([], [], 'o-', color=aqua, alpha=.1, label="Closest Road Succ", zorder=3)
            self.closest_road_succs.append(s)

        self.closest_lane_preds = []
        self.closest_lane_neighbors = []
        for i in range(MAX_PREDS):
            p, *_ = self.ax.plot([], [], 'o-', color=light_orange,  alpha=.1, label="Closest Lane Pred", zorder=3)
            self.closest_lane_preds.append(p)
        for i in range(MAX_SUCCS):
            s, *_ = self.ax.plot([], [], 'o-', color=light_aqua,  alpha=.1, label="Closest Lane Pred", zorder=3)
            self.closest_lane_neighbors.append(s)

        #################
        # Texts
        self.road_id_text = self.ax.text(0, 0, '', fontsize=12, color=red, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.road_succ_id_text = self.ax.text(0, 0, '', fontsize=10, color=aqua, alpha=0.8, zorder=4, ha='center', va='center')
        self.road_pred_id_text = self.ax.text(0, 0, '', fontsize=10, color=orange, alpha=0.8, zorder=4, ha='center', va='center')
        self.lane_id_text = self.ax.text(0, 0, '', fontsize=12, color=yellow, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.junct_id_text = self.ax.text(0, 0, '', fontsize=10, color=red, alpha=0.8, zorder=4, ha='center', va='center')
        
        self.lane_succ_id_text = self.ax.text(0, 0, '', fontsize=8, color=light_aqua, alpha=0.8, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.lane_pred_id_text = self.ax.text(0, 0, '', fontsize=8, color=orange, alpha=0.8, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        #################

        self.__tmp_road, *_ = self.ax.plot([], [], 'o-', color='blue', markersize=5, alpha=.1, label="Closest Road", zorder=3)

        self.lane_path_plots = []
        for i in range(MAX_ROAD_PATH):
           pl , *_ = self.ax.plot([], [], 'o-', color=blue, linewidth=2, alpha=0.5, label="Lane Path", zorder=2)
           self.lane_path_plots.append(pl)

        self.global_plan_path = LineCollection([], cmap=_VELOCITY_CMAP, linewidths=5, zorder=4)
        self.ax.add_collection(self.global_plan_path)

        self.road_arrow = None # for the road direction arrow
        self.lane_arrow = None # for the lane direction arrow

    def plot(self, exec: SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True,
             follow_vehicle=True, show_plan_boundaries: bool = True,
             velocity_scale: str = "relative",
             delta: Optional[tuple[float, float]] = None):
        self._velocity_scale = velocity_scale
        if not self.map_plotted:
            self.plot_map(exec.global_planner)
        self.plot_vehicle(exec.ego_state)
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)
        plan = exec.global_planner.global_plan
        if len(plan.path) < 2:
            plan = exec.local_planner.global_plan
        if len(plan.path) >= 2:
            path = np.array(plan.path).T
            velocity = plan.velocity or plan.trajectory.velocity
            _update_velocity_colored_line(
                self.global_plan_path, path[0], path[1], velocity,
                velocity_scale=self._velocity_scale,
            )
        self.fig.canvas.draw()

    def show_closest_road_and_lane(self,  x:int, y:int, map:HDMap):
        """Show the closest road and lane to the given coordinates"""
        l = map.find_nearest_lane(x,y)

        if l is not None:
            # log.debug(f"Lane ID: {l.id}, Road: {l.road_id} Lane Type: {l.type}")
            self.__clear_closest_road_and_lane()
            self.closest_lane.set_data(l.center_line[0], l.center_line[1])
            line_xs = l.center_line[0]
            line_ys = l.center_line[1]
            if int(l.id) < 0: 
                lx1, lx2 = line_xs[-2], line_xs[-1]
                ly1, ly2 = line_ys[-2], line_ys[-1]
            else:
                lx1, lx2 = line_xs[1], line_xs[0]
                ly1, ly2 = line_ys[1], line_ys[0]
            self.lane_arrow = self.ax.annotate('', xy=(lx2, ly2), xytext=(lx1, ly1), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color=self.yellow, lw=2), zorder=5)
            # log.debug(f"lane p1: ({lx1}, {ly1}), p2: ({lx2}, {ly2})")
            center_idx = int(len(l.center_line[0]) / 2.5)
            self.lane_id_text.set_position((l.center_line[0][center_idx], l.center_line[1][center_idx]-5))
            self.lane_id_text.set_text(f"{l.id}")
            self.lane_pred_id_text.set_position((l.center_line[0][0], l.center_line[1][0]-5))
            self.lane_pred_id_text.set_text(f"P: {l.pred_id}")

            self.lane_succ_id_text.set_position((l.center_line[0][-1], l.center_line[1][-1]-5))
            self.lane_succ_id_text.set_text(f"S: {l.succ_id}")
            for i,s in enumerate(l.neighbors):
                self.closest_lane_neighbors[i].set_data(s.center_line[0], s.center_line[1])

            # if l.type == "driving":
            #     log.debug(f"Neighbors({l.uid}): {[n.uid for n in l.neighbors]}")
            #     for n in l.neighbors:
            #         log.debug(f" Can I go to {n.uid}? {map.can_laneA_access_laneB(l,n)}")


            r:HDMap.Road|None = map.road_by_id.get(l.road_id)
            if r is not None:
                self.closest_road.set_data(r.center_line[0], r.center_line[1])
                x1, x2 = r.center_line[0][-2], r.center_line[0][-1]
                y1, y2 = r.center_line[1][-2], r.center_line[1][-1]
                self.road_arrow = self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color=self.red, lw=2), zorder=5)

                center_idx = int(len(r.center_line[0]) / 2)
                self.road_id_text.set_position((r.center_line[0][center_idx], r.center_line[1][center_idx]))
                self.road_id_text.set_text(r.id)
                self.junct_id_text.set_position((r.center_line[0][center_idx], r.center_line[1][center_idx]-5))
                self.junct_id_text.set_text(f"{r.junction_id}+") if r.junction_id != "-1" else self.junct_id_text.set_text("")
                self.road_pred_id_text.set_position((r.center_line[0][0]-3, r.center_line[1][0]))
                p_txt = f"P: {r.pred_id}" if r.pred_type == "road" else f"P: {r.pred_id}+"
                self.road_pred_id_text.set_text(p_txt)
                self.road_succ_id_text.set_position((r.center_line[0][-1]+3, r.center_line[1][-1]))
                s_txt = f"S: {r.succ_id}" if r.succ_type == "road" else f"S: {r.succ_id}+"
                self.road_succ_id_text.set_text(s_txt)
                for i,s in enumerate(r.successors):
                    self.closest_road_succs[i].set_data(s.center_line[0], s.center_line[1])
                for i,p in enumerate(r.predecessors):
                    self.closest_road_preds[i].set_data(p.center_line[0], p.center_line[1])
                # log.debug(f"getting connecting roads: {[p for p in map._get_connecting_roads_from_junction(map.root, r.road_element, r.pred_id )]}")
                # log.debug(f"Road preds: {[p.id for p in r.predecessors]}, succs: {[s.id for s in r.successors]}")
                # log.debug(f"Road ID: {r.id}, lane sections: {[l[0].id for s,l in r.lane_sections.items()]}")
        
        
        self.fig.canvas.draw()

    def plot_global_plan(self, global_plan: GlobalPlan):
        """Plot the road path"""
        try:
            log.debug("Plotting Road Path: length = %d", len(global_plan.path))
            self.clear_road_path_plots()
            path = np.array(global_plan.path).T
            velocity = global_plan.velocity or global_plan.trajectory.velocity
            _update_velocity_colored_line(
                self.global_plan_path, path[0], path[1], velocity,
                velocity_scale=self._velocity_scale,
            )
            self.fig.canvas.draw()
        except Exception as e:
            log.error(f"Error plotting global plan: {e}")
            _update_velocity_colored_line(self.global_plan_path, [], [], [])


    def clear_road_path_plots(self):
        """Clear the road path plots"""
        for i in range(len(self.lane_path_plots)):
            self.lane_path_plots[i].set_data([], [])
        _update_velocity_colored_line(self.global_plan_path, [], [], [])
        self.__clear_closest_road_and_lane()

    
    def clear_tmp_plots(self):
        super().clear_tmp_plots()
        self.__clear_closest_road_and_lane()

    def __clear_closest_road_and_lane(self):
        """Clear the closest road and lane plots"""
        self.closest_road.set_data([], [])
        self.closest_lane.set_data([], [])
        self.road_id_text.set_text("")
        self.junct_id_text.set_text("")
        self.road_pred_id_text.set_text("")
        self.road_succ_id_text.set_text("")
        self.lane_id_text.set_text("")
        self.lane_pred_id_text.set_text("")
        self.lane_succ_id_text.set_text("")
        
        for pr in self.closest_road_preds:
            pr.set_data([],[])
        for sr in self.closest_road_succs:
            sr.set_data([],[])
        for pl in self.closest_lane_preds:
            pl.set_data([],[])
        for sl in self.closest_lane_neighbors:
            sl.set_data([],[])

        if self.road_arrow is not None:
            self.road_arrow.remove()
            self.road_arrow = None
        if self.lane_arrow is not None:
            self.lane_arrow.remove()
            self.lane_arrow = None

        self.fig.canvas.draw()

    def clear_goal(self):
        """Clear the goal plot"""
        self.goal.set_data([], [])
        self.goal_text.set_text("")
        self.fig.canvas.draw()
        
    def plot_map(self, global_planner:GlobalPlannerStrategy, show_road_points=False, show_lane_points=False):
        """Implement the abstract method from GlobalPlot"""
        
        if not hasattr(global_planner, "hdmap"):
            log.warning("HDMap not found in the global planner.")
            return

        global_planner = cast(HDMapGlobalPlanner,global_planner)
        hdmap = global_planner.hdmap

        #TODO: remove this later if not needed
        if show_road_points:
            map = global_planner.hdmap
            p = np.array(list(map.point_to_road.keys())).T
            log.debug(f"road points length {len(p)}")
            self.ax.scatter(p[0], p[1], color='blue', s=10, alpha=0.5)
        if show_lane_points:
            map = global_planner.hdmap
            p = np.array(list(map.point_to_lane.keys())).T
            log.debug(f"lane points length {len(p)}")
            self.ax.scatter(p[0], p[1], color='blue', s=10, alpha=0.5)


        all_x_coords = []
        all_y_coords = []

        # for r in hdmap.roads:
            # self.ax.plot(r.center_line[0], r.center_line[1], color='red', linewidth=2, alpha=0.5)

        for l in hdmap.lanes:
            if l.type == "driving":
                color = "#427b58" 
                alpha = 0.7
            elif l.type == "shoulder":
                color = "#af3a03"
                alpha = 0.4
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
    def __init__(self, max_plan_length=5, max_agent_count=12, show_occupancy_flow=True, occupancy_flow_shape=(100, 100), controller: Optional[ControlStrategy] = None):
        self.MAX_PLAN_LENGTH = max_plan_length
        self.MAX_AGENT_COUNT = max_agent_count
        self.controller = controller

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        # self.ax2.set_title("Frenet Coordinate", pad=-100)

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
        self.view_width_ax1 = None
        self.view_height_ax1 = None
        self.view_width_ax2 = None
        self.view_height_ax2 = None
        
        self.orientation_arrow = None  # For the vehicle orientation arrow
        

        for i in range(self.MAX_PLAN_LENGTH):
            (local_plan_ax1,) = self.ax1.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
            (local_plan_ax2,) = self.ax2.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
            self.local_plan_plots_ax1.append(local_plan_ax1)
            self.local_plan_plots_ax2.append(local_plan_ax2)

        (self.left_boundry_x1,) = self.ax1.plot([], [], color="orange", label="Left Boundary", linewidth=2)
        (self.right_boundry_x1,) = self.ax1.plot([], [], color="tan", label="Right Boundary", linewidth=2)
        (self.left_boundry_ax2,) = self.ax2.plot([], [], color="orange", linewidth=1, label="Left Boundary (Ref)")
        (self.right_boundry_ax2,) = self.ax2.plot([], [], color="tan", linewidth=1, label="Right Boundary (Ref)")

        (self.reference_trajectory_ax1,) = self.ax1.plot([], [], "gray", label="Reference Trajectory", linewidth=2)
        (self.reference_trajectory_ax2,) = self.ax2.plot(
            [], [], color="gray", linewidth=1, alpha=0.5, label="Global Trajectory"
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

        (self.car_heading_plot,) = self.ax1.plot([], [], "-", color="darkslategray", label="Car Heading")
        (self.car_location_plot,) = self.ax1.plot([], [], "ko", markersize=7, label="Car Location")

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

        # LiDAR point cloud scatter (XY view on ax1, Frenet S-D view on ax2)
        self.lidar_scatter_ax1 = self.ax1.scatter([], [], s=6, c='lime', alpha=0.9, zorder=6, label="LiDAR")
        self.lidar_scatter_ax2 = self.ax2.scatter([], [], s=6, c='lime', alpha=0.9, zorder=6, label="LiDAR")

        # Clustered LiDAR points that passed segmentation + range gating (diagnostic, XY + Frenet)
        self.cluster_scatter_ax1 = self.ax1.scatter([], [], s=8, c='yellow', alpha=0.9, zorder=7, label="Clusters")
        self.cluster_scatter_ax2 = self.ax2.scatter([], [], s=8, c='yellow', alpha=0.9, zorder=7, label="Clusters")

        # Track boundary segments from the worldbridge (e.g. BasicSim.boundary_segments)
        self.track_boundary_collection = LineCollection(
            [], linewidths=1, colors='cyan', alpha=0.5, zorder=2, label="Track Boundary"
        )
        self.ax1.add_collection(self.track_boundary_collection)
        self.track_boundary_collection_ax2 = LineCollection(
            [], linewidths=1, colors='cyan', alpha=0.5, zorder=2
        )
        self.ax2.add_collection(self.track_boundary_collection_ax2)

        # Prediction trajectories: one dotted orange polyline per agent on both views
        self.prediction_lines_ax1 = []
        self.prediction_lines_ax2 = []
        for _ in range(self.MAX_AGENT_COUNT):
            l1, = self.ax1.plot([], [], color="darkorange", linewidth=1.5, linestyle="dotted", zorder=3)
            l2, = self.ax2.plot([], [], color="darkorange", linewidth=1.5, linestyle="dotted", zorder=3)
            self.prediction_lines_ax1.append(l1)
            self.prediction_lines_ax2.append(l2)

        # self.pm_occupancy_flow_ax1 = self.ax1.imshow(
        #         np.zeros((100, 100)),
        #         origin='upper',
        #         extent=[
        #             pm.grid_bounds.get('min_x', 0),
        #             pm.grid_bounds.get('max_x', 0) + 100 * pm.grid_bounds.get('resolution', 1), 
        #             pm.grid_bounds.get('min_y', 0),
        #             pm.grid_bounds.get('max_y', 0) + 100 * pm.grid_bounds.get('resolution', 1)
        #         ]
        #     )

        self.legend_ax = self.fig.add_axes([0.0, -0.013, 1, 0.1])
        self.legend_ax.legend(
            *self.ax1.get_legend_handles_labels(), loc="center", ncol=7, borderaxespad=0.0, fontsize=7, framealpha=0.3)
        for tick in self.ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
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
        frenet_follow_planner = False,
        plot_occupancy_flow = False,
        plot_predictions = True,
        plot_lidar = False,
        lidar_data = None,
        plot_lidar_global = True,
        plot_lidar_frenet = False,
        plot_clusters = True,
        plot_ground_truth = True,
        show_global_view = True,
        show_frenet_view = True,
        plot_race_boundary = True,
    ):
        self.legend_ax.set_visible(show_legend)
        self.ax1.set_visible(show_global_view)
        self.ax2.set_visible(show_frenet_view)

        # Adjust GridSpec height ratios so the visible axis fills the available space.
        # hspace=0 eliminates the inter-row gap when only one axis is visible.
        _gs = self.ax1.get_subplotspec().get_gridspec()
        if show_global_view and show_frenet_view:
            _gs._row_height_ratios = [1, 1]
            self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)
        elif show_global_view:
            _gs._row_height_ratios = [1, 0.001]
            self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1, hspace=0)
        elif show_frenet_view:
            _gs._row_height_ratios = [0.001, 1]
            self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1, hspace=0)

        center_xy = exec.local_planner.location_xy if global_follow_planner else  (exec.ego_state.x, exec.ego_state.y)
        center_sd = exec.local_planner.location_sd if frenet_follow_planner else \
            exec.local_planner.global_trajectory.convert_xy_to_sd(exec.ego_state.x, exec.ego_state.y)
        if xy_zoom is not None:
            self.ax1.set_xlim(center_xy[0] - xy_zoom, center_xy[0] + xy_zoom)
            self.ax1.set_ylim( center_xy[1] - xy_zoom / aspect_ratio / 2, center_xy[1] + xy_zoom / aspect_ratio / 2,)
            self.view_width_ax1 = xy_zoom * 2
            self.view_height_ax1 = xy_zoom / aspect_ratio * 2
        if frenet_zoom is not None:
            self.ax2.set_xlim(center_sd[0] - frenet_zoom, center_sd[0] + frenet_zoom)
            self.view_width_ax2 = frenet_zoom * 2
            self.ax2.set_ylim(-frenet_zoom / aspect_ratio / 2, frenet_zoom / aspect_ratio / 2)
            self.view_height_ax2 = frenet_zoom / aspect_ratio * 2

        if plot_last_pts and num_plot_last_pts > 0:
            self.last_locs_ax1.set_data( exec.local_planner.traversed_x[-num_plot_last_pts:], exec.local_planner.traversed_y[-num_plot_last_pts:])
            self.planner_loc_ax1.set_data([exec.local_planner.location_xy[0]], [exec.local_planner.location_xy[1]])

            self.last_locs_ax2.set_data( exec.local_planner.traversed_s[-num_plot_last_pts:], exec.local_planner.traversed_d[-num_plot_last_pts:])
            self.planner_loc_ax2.set_data([exec.local_planner.location_sd[0]], [exec.local_planner.location_sd[1]])
        else:
            self.last_locs_ax1.set_data([], [])
            self.planner_loc_ax1.set_data([], [])
            self.last_locs_ax2.set_data([], [])
            self.planner_loc_ax2.set_data([], [])

        show_race_boundaries = plot_race_boundary and not isinstance(
            exec.global_planner, HDMapGlobalPlanner
        )
        fallback_plan = getattr(exec.global_planner, "global_plan", None)
        self.update_track_boundary_plot(
            exec.world,
            show_plot=show_race_boundaries,
            global_trajectory=exec.local_planner.global_trajectory,
            fallback_plan=fallback_plan,
        )
        self.update_global_plan_plots(exec.local_planner, plot_global_plan)
        self.update_lattice_graph_plots(exec.local_planner, plot_local_lattice)
        self.update_local_plan_plots(exec.local_planner, plot_local_plan)
        self.update_state_plots(exec.ego_state, exec.local_planner.global_trajectory, plot_state)
        exec_pm = exec.pm
        world_pm = (
            exec.world.get_ground_truth_perception_model()
            if plot_ground_truth and hasattr(exec.world, "get_ground_truth_perception_model")
            else None
        )
        self.update_perception_model_plots(
            exec_pm,
            exec.local_planner.global_trajectory,
            plot_perception_model,
            plot_predictions,
            world_pm=world_pm,
        )
        self.update_lidar_plot(lidar_data, plot_lidar, exec.local_planner.global_trajectory, plot_lidar_global, plot_lidar_frenet)
        self.update_cluster_plot(getattr(exec.pm, "detection_clusters", None), plot_clusters, exec.local_planner.global_trajectory, plot_lidar_frenet)
        self.update_pm_occupancy_flow_plots(exec.pm, plot_occupancy_flow)

    def update_track_boundary_plot(
        self,
        world_bridge,
        show_plot=True,
        global_trajectory=None,
        fallback_plan=None,
    ):
        if not show_plot or not hasattr(world_bridge, 'boundary_segments'):
            self.track_boundary_collection.set_segments([])
            self.track_boundary_collection_ax2.set_segments([])
            return
        segs = world_bridge.boundary_segments  # (M, 2, 2) world-frame
        if len(segs) == 0 and fallback_plan is not None:
            from avlite.c40_execution.c46_basic_sim import boundary_segments_from_global_plan

            segs = boundary_segments_from_global_plan(fallback_plan)
        self.track_boundary_collection.set_segments(segs if len(segs) else [])
        if global_trajectory is None or len(segs) == 0:
            self.track_boundary_collection_ax2.set_segments([])
            return
        # Convert all segment endpoints to Frenet (s, d) and draw on ax2.
        # Filter out segments that span across the lap start/finish (wrap-around
        # produces a huge s-difference and causes long diagonal lines in Frenet view).
        pts = segs.reshape(-1, 2)                          # (M*2, 2)
        sd  = global_trajectory.convert_xy_path_to_sd_path_np(pts)  # (M*2, 2)
        sd_segs = sd.reshape(-1, 2, 2)                     # (M, 2, 2)
        s_diff = np.abs(sd_segs[:, 1, 0] - sd_segs[:, 0, 0])
        valid = s_diff < (np.median(s_diff) * 20 + 50.0)
        self.track_boundary_collection_ax2.set_segments(sd_segs[valid] if valid.any() else [])

    def redraw_plots(self):
        self.ax1.draw_artist(self.ax1.patch)
        self.ax2.draw_artist(self.ax2.patch)

        for line in self.ax1.lines:
            self.ax1.draw_artist(line)
        for line in self.ax2.lines:
            self.ax2.draw_artist(line)

        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)

    def update_global_plan_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not show_plot:
            self.g_wp_current_ax1.set_data([], [])
            self.g_wp_current_ax2.set_data([], [])
            self.g_wp_next_ax1.set_data([], [])
            self.g_wp_next_ax2.set_data([], [])
            self.left_boundry_x1.set_data([], [])
            self.right_boundry_x1.set_data([], [])
            self.left_boundry_ax2.set_data([], [])
            self.right_boundry_ax2.set_data([], [])
            self.reference_trajectory_ax1.set_data([], [])
            self.reference_trajectory_ax2.set_data([], [])
            return

        if not pl.global_plan.left_boundary_x or not pl.global_plan.right_boundary_x:
            self.left_boundry_x1.set_data([], [])
            self.right_boundry_x1.set_data([], [])
            self.left_boundry_ax2.set_data([], [])
            self.right_boundry_ax2.set_data([], [])
        else:
            self.left_boundry_x1.set_data(pl.global_plan.left_boundary_x, pl.global_plan.left_boundary_y)
            self.right_boundry_x1.set_data(pl.global_plan.right_boundary_x, pl.global_plan.right_boundary_y)
        _s = np.asarray(pl.global_trajectory.path_s, dtype=float)
        _gaps = np.where(np.diff(_s) < 0)[0] + 1
        if pl.global_plan.left_boundary_d and pl.global_plan.right_boundary_d:
            if len(_gaps):
                _s   = np.insert(_s,   _gaps, np.nan)
                _ld  = np.insert(np.asarray(pl.global_plan.left_boundary_d,  dtype=float), _gaps, np.nan)
                _rd  = np.insert(np.asarray(pl.global_plan.right_boundary_d, dtype=float), _gaps, np.nan)
                _ref = np.insert(np.asarray(pl.global_trajectory.path_d,     dtype=float), _gaps, np.nan)
            else:
                _ld  = pl.global_plan.left_boundary_d
                _rd  = pl.global_plan.right_boundary_d
                _ref = pl.global_trajectory.path_d
            self.left_boundry_ax2.set_data(_s, _ld)
            self.right_boundry_ax2.set_data(_s, _rd)
            self.reference_trajectory_ax2.set_data(_s, _ref)
        else:
            self.left_boundry_ax2.set_data([], [])
            self.right_boundry_ax2.set_data([], [])
            if len(_gaps):
                _s = np.insert(_s, _gaps, np.nan)
                _ref = np.insert(np.asarray(pl.global_trajectory.path_d, dtype=float), _gaps, np.nan)
            else:
                _ref = pl.global_trajectory.path_d
            self.reference_trajectory_ax2.set_data(_s, _ref)
        self.reference_trajectory_ax1.set_data(pl.global_trajectory.path_x, pl.global_trajectory.path_y)

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

    def update_lattice_graph_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not show_plot or not hasattr(pl, "lattice") or len(pl.lattice.edges) == 0:
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
            if edge_index >= len(self.lattice_graph_plots_ax1):
                (line_ax1,) = self.ax1.plot([], [], "--", color="lightskyblue", alpha=0.6)
                (line_ax2,) = self.ax2.plot([], [], "--", color="lightskyblue", alpha=0.6)
                (endpoint_ax1,) = self.ax1.plot([], [], "bo", alpha=0.6)
                (endpoint_ax2,) = self.ax2.plot([], [], "bo", alpha=0.6)
                self.lattice_graph_plots_ax1.append(line_ax1)
                self.lattice_graph_plots_ax2.append(line_ax2)
                self.lattice_graph_endpoints_ax1.append(endpoint_ax1)
                self.lattice_graph_endpoints_ax2.append(endpoint_ax2)


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

        for i in range(edge_index, len(self.lattice_graph_plots_ax1)):
            self.lattice_graph_plots_ax1[i].set_data([], [])
            self.lattice_graph_plots_ax2[i].set_data([], [])
            self.lattice_graph_endpoints_ax1[i].set_data([], [])
            self.lattice_graph_endpoints_ax2[i].set_data([], [])

    def update_local_plan_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not hasattr(pl, "selected_local_plan"):
            # Non-lattice planners expose a single LocalPlan trajectory.
            self.current_wp_plot_ax1.set_data([], [])
            self.current_wp_plot_ax2.set_data([], [])
            self.next_wp_plot_ax1.set_data([], [])
            self.next_wp_plot_ax2.set_data([], [])
            local_plan = pl.get_local_plan() if show_plot else None
            tj = local_plan.as_trajectory() if local_plan is not None else None
            if tj is None:
                self.__clear_local_plan_plots()
            else:
                self.local_plan_plots_ax1[0].set_data(tj.path_x, tj.path_y)
                s_plot, d_plot = pl.global_trajectory.convert_xy_path_to_sd_path(list(zip(tj.path_x, tj.path_y)))
                self.local_plan_plots_ax2[0].set_data(list(s_plot), list(d_plot))
                self.__clear_local_plan_plots(index=1)
            return
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

    def update_state_plots(self, state: EgoState, global_trajectory: TrajectoryTracker, show_plot=True):
        if not show_plot:
            self.car_heading_plot.set_data([], [])
            self.car_location_plot.set_data([], [])
            self.ego_vehicle_ax1.set_xy(np.empty((0, 2)))
            self.ego_vehicle_ax2.set_xy(np.empty((0, 2)))
            return


        car_L_f = self.controller.ego_distance_front_axle if self.controller is not None else 2.5
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

    def update_perception_model_plots(
        self,
        exec_pm: PerceptionModel,
        global_trajectory: TrajectoryTracker,
        show_plot=True,
        show_prediction=False,
        world_pm: Optional[PerceptionModel] = None,
    ):
        pm_agents = world_pm if world_pm is not None else exec_pm
        if not show_plot or len(pm_agents.agent_vehicles) == 0:
            for i in range(self.MAX_AGENT_COUNT):
                self.pm_plots_ax1[i].set_xy(np.empty((0, 2)))
                self.pm_plots_ax2[i].set_xy(np.empty((0, 2)))
                self.prediction_lines_ax1[i].set_data([], [])
                self.prediction_lines_ax2[i].set_data([], [])
            return

        n = min(len(pm_agents.agent_vehicles), self.MAX_AGENT_COUNT)
        if len(pm_agents.agent_vehicles) > self.MAX_AGENT_COUNT:
            log.warning(f"Exceeded maximum number of agents: {self.MAX_AGENT_COUNT}")
        agents = pm_agents.agent_vehicles[:n]
        # Batch all corners into one KD-tree call (n*4 points) instead of 4 calls per agent
        all_corners_xy = np.vstack([agent.get_bb_corners() for agent in agents])  # (n*4, 2)
        all_corners_sd = global_trajectory.convert_xy_path_to_sd_path_np(all_corners_xy)  # (n*4, 2)
        for i, agent in enumerate(agents):
            self.pm_plots_ax1[i].set_xy(agent.get_bb_corners())
            self.pm_plots_ax2[i].set_xy(all_corners_sd[i * 4:(i + 1) * 4])
        # clear stale patches left over from the previous frame when agent count drops
        for j in range(n, self.MAX_AGENT_COUNT):
            self.pm_plots_ax1[j].set_xy(np.empty((0, 2)))
            self.pm_plots_ax2[j].set_xy(np.empty((0, 2)))

        # Prediction trajectories (always from exec_pm; world_pm has no pipeline outputs)
        use_prediction = (
            show_prediction
            and exec_pm.prediction_mode == PredictionMode.TRAJECTORY
            and exec_pm.trajectories is not None
            and exec_pm.trajectories.shape[0] >= n
        )
        if use_prediction:
            ego_hdg = np.array([np.cos(exec_pm.ego_vehicle.theta), np.sin(exec_pm.ego_vehicle.theta)])
            for i, agent in enumerate(agents):
                to_agent = np.array([agent.x - exec_pm.ego_vehicle.x, agent.y - exec_pm.ego_vehicle.y])
                if float(np.dot(ego_hdg, to_agent)) < 0.0:
                    self.prediction_lines_ax1[i].set_data([], [])
                    self.prediction_lines_ax2[i].set_data([], [])
                    continue
                path_xy = np.vstack([[agent.x, agent.y], exec_pm.trajectories[i]])
                self.prediction_lines_ax1[i].set_data(path_xy[:, 0], path_xy[:, 1])
                path_sd = global_trajectory.convert_xy_path_to_sd_path_np(path_xy)
                self.prediction_lines_ax2[i].set_data(path_sd[:, 0], path_sd[:, 1])
        for i in range(n if use_prediction else 0, self.MAX_AGENT_COUNT):
            self.prediction_lines_ax1[i].set_data([], [])
            self.prediction_lines_ax2[i].set_data([], [])

    def update_lidar_plot(self, lidar_data, show_plot=True, global_trajectory=None, show_global=True, show_frenet=False):
        """Update the LiDAR scatter on ax1 (XY view) and ax2 (Frenet S-D view)."""
        if not show_plot or lidar_data is None or len(lidar_data) == 0:
            self.lidar_scatter_ax1.set_offsets(np.empty((0, 2)))
            self.lidar_scatter_ax2.set_offsets(np.empty((0, 2)))
            return
        # lidar_data is (N,2+) [x,y,...] – plot X,Y on the global view
        xy = lidar_data[:, :2]
        # Global (XY) view, gated by show_global
        if show_global:
            self.lidar_scatter_ax1.set_offsets(xy)
        else:
            self.lidar_scatter_ax1.set_offsets(np.empty((0, 2)))
        # Frenet view: convert world points to (s, d), gated by show_frenet
        if show_frenet and global_trajectory is not None:
            self.lidar_scatter_ax2.set_offsets(global_trajectory.convert_xy_path_to_sd_path_np(xy))
        else:
            self.lidar_scatter_ax2.set_offsets(np.empty((0, 2)))

    def update_cluster_plot(self, clusters, show_plot=True, global_trajectory=None, show_frenet=False):
        """Highlight (XY + Frenet) the LiDAR points that passed segmentation + range gating."""
        if not show_plot or clusters is None or len(clusters) == 0:
            self.cluster_scatter_ax1.set_offsets(np.empty((0, 2)))
            self.cluster_scatter_ax2.set_offsets(np.empty((0, 2)))
            return
        xy = np.asarray(clusters)[:, :2]
        self.cluster_scatter_ax1.set_offsets(xy)
        if show_frenet and global_trajectory is not None:
            self.cluster_scatter_ax2.set_offsets(global_trajectory.convert_xy_path_to_sd_path_np(xy))
        else:
            self.cluster_scatter_ax2.set_offsets(np.empty((0, 2)))

    def update_pm_occupancy_flow_plots(self, pm: Optional[PerceptionModel]=None, show_plot=True):
        if not show_plot or pm is None:
            if hasattr(self, 'pm_occupancy_flow_ax1'):
                self.pm_occupancy_flow_ax1.set_data(np.zeros((100, 100)))
                self.pm_occupancy_flow_ax1.set_extent([0, 0, 0, 0])
            return
        if pm.occupancy_flow is not None and pm.grid_bounds is not None:
            extent = [
                pm.grid_bounds.get('min_x', 0),
                pm.grid_bounds.get('max_x', 0), 
                pm.grid_bounds.get('min_y', 0),
                pm.grid_bounds.get('max_y', 0),
            ]
            flow_sum = pm.occupancy_flow[0].T
            if not hasattr(self, 'pm_occupancy_flow_ax1'):
                flow_sum = np.sum(pm.occupancy_flow, axis=0)
                self.pm_occupancy_flow_ax1 = self.ax1.imshow(
                    flow_sum,
                    origin='lower',
                    extent=extent,
                    cmap='plasma',
                    vmin=0,
                    vmax=1
                )
                # self.fig.colorbar(self.pm_occupancy_flow_ax1, ax=self.ax1, label='Occupancy')
            else:
                self.pm_occupancy_flow_ax1.set_data(flow_sum)
                self.pm_occupancy_flow_ax1.set_extent(extent)
            self.fig.canvas.draw_idle()


    def set_plot_theme(self, bg_color="white", fg_color="black"):
        self.fig.patch.set_facecolor(bg_color)
        self.ax1.patch.set_facecolor(bg_color)
        self.ax2.patch.set_facecolor(bg_color)
        # self.ax2.set_title("Frenet Coordinate", color=fg_color)
        # Set titles and labels to white
        for ax in [self.ax1, self.ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor(fg_color)
            ax.tick_params(axis="both", colors=fg_color)  # Set tick colors to white
            ax.xaxis.label.set_color(fg_color)  # Set x-axis label color to white
            ax.yaxis.label.set_color(fg_color)  # Set y-axis label color to white

        log.debug(f"Plot theme set to {bg_color} background and {fg_color} foreground.")
        self.redraw_plots()
    
    def show_vehicle_orientation_ax1(self, x, y, theta):
        """Show the vehicle orientation on the plot"""
        log.debug(f"Showing vehicle orientation at ({x}, {y}) with theta={theta}")

        length = .15 * min(self.view_width_ax1, self.view_height_ax1) if self.view_height_ax1 is not None or self.view_width_ax1 is None else 20

        x2 = x + length * np.cos(theta)
        y2 = y + length * np.sin(theta)
        if self.orientation_arrow:
            self.orientation_arrow.remove()
        self.orientation_arrow = self.ax1.annotate('', xy=(x2, y2), xytext=(x,y), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color="red", lw=5), zorder=5)

    def show_vehicle_orientation_ax2(self, s, d, theta):
        """Show the vehicle orientation on the plot"""
        log.debug(f"Showing vehicle orientation at ({s}, {d}) with theta={theta}")

        length = .15 * min(self.view_width_ax2, self.view_height_ax2) if self.view_height_ax2 is not None or self.view_width_ax2 is None else 20
             

        s2 = s + length * np.cos(theta)
        d2 = d + length * np.sin(theta)
        if self.orientation_arrow:
            self.orientation_arrow.remove()
        self.orientation_arrow = self.ax2.annotate('', xy=(s2, d2), xytext=(s,d), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color="red", lw=5), zorder=5)

    def clear_tmp_plots(self):
        self.orientation_arrow.remove() if self.orientation_arrow else None
        self.orientation_arrow = None

    def reset(self):
        self.update_pm_occupancy_flow_plots(None, show_plot=False)


# --- module helpers ---

# slow → fast : green → yellow → red (distinct from HD lane #427b58)
_VELOCITY_CMAP = LinearSegmentedColormap.from_list(
    "velocity_slow_fast",
    [(0.0, "#39ff14"), (0.5, "#ffd700"), (1.0, "#dc143c")],
)


def _update_velocity_colored_line(collection, x, y, velocity, *, velocity_scale="relative"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        collection.set_segments([])
        collection.set_array(np.array([]))
        return
    v = np.asarray(velocity, float)
    if len(v) == 0:
        v = np.full(len(x), 0.5)
    n = min(len(x), len(v))
    x, y, v = x[:n], y[:n], v[:n]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_v = 0.5 * (v[:-1] + v[1:])
    collection.set_segments(segments)
    collection.set_array(seg_v)
    if velocity_scale == "absolute":
        vmax = float(ControlSettings.c32_ego_max_velocity)
        if vmax <= 0.0:
            vmax = 1e-9
        collection.set_norm(Normalize(vmin=0.0, vmax=vmax, clip=True))
    else:
        vmin, vmax = float(seg_v.min()), float(seg_v.max())
        if vmin == vmax:
            vmax = vmin + 1e-9
        collection.set_norm(Normalize(vmin=vmin, vmax=vmax))
