import math
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline
import logging

from c20_planning.c22_base_global_planner import BaseGlobalPlanner
from c20_planning.c21_planning_model import GlobalPlan

log = logging.getLogger(__name__)

class GlobalHDMapPlanner(BaseGlobalPlanner):
    """
    A global planner that:
      1. Parses a simplified OpenDRIVE (.xodr) file.
      2. Builds a lane-center graph by sampling the parametric road geometry.
      3. Uses A* to find a path from start to goal.
      4. Returns a smooth path and a simple velocity profile.
    """

    def __init__(self, xodr_file, sampling_resolution=1.0):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution
        try:
            tree = ET.parse(self.xodr_file)
            self.xodr_root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing OpenDRIVE file: {e}")



    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass

    def __generate_lane_graph(self):
        self.graph = nx.DiGraph()
        raise NotImplementedError("Lane graph generation not implemented yet.")

    def __plan_global_path(source, destination):
        # 1. Road-level planning
        road_path = find_shortest_road_path(source, destination)
        
        # 2. Lane-level planning
        lane_path = []
        for i in range(len(road_path) - 1):
            current_road = road_path[i]
            next_road = road_path[i+1]
            
            # Find valid lanes for current road segment
            valid_lanes = get_valid_driving_lanes(current_road)
            
            # Find lanes that connect to the next road
            connecting_lanes = filter_connecting_lanes(valid_lanes, current_road, next_road)
            
            # Choose optimal lane based on minimizing lane changes
            optimal_lane = select_optimal_lane(connecting_lanes, lane_path[-1] if lane_path else None)
            lane_path.append(optimal_lane)
        
        return lane_path


def sample_OpenDrive_geometry(x0, y0, hdg, length, geom_type='line', attributes=None, n_pts=50):
        """
        Returns (x_vals, y_vals) for various geometry types in OpenDRIVE.
        Supports line, arc, and basic handling for other types.
        """
        x_vals, y_vals = [], []
        s_array = np.linspace(0, length, n_pts)

        ### Arc case
        if geom_type == 'arc' and attributes is not None:
            curvature = float(attributes.get('curvature', 0))
            if curvature != 0:
                radius = abs(1.0 / curvature)  # Use absolute value for radius
                # Determine arc direction based on curvature sign
                arc_direction = np.sign(curvature)
                
                # Calculate center of the arc
                # For positive curvature (left turn), center is to the left of heading
                # For negative curvature (right turn), center is to the right of heading
                center_x = x0 - np.sin(hdg) * radius * arc_direction
                center_y = y0 + np.cos(hdg) * radius * arc_direction
                
                # Calculate start angle (from center to start point)
                start_angle = np.arctan2(y0 - center_y, x0 - center_x)
                
                # Calculate angle change over the arc length
                dtheta = length / radius * arc_direction
                
                # Generate points along the arc
                angles = np.linspace(start_angle, start_angle + dtheta, n_pts)
                for angle in angles:
                    x_vals.append(center_x + radius * np.cos(angle))
                    y_vals.append(center_y + radius * np.sin(angle))
                return x_vals, y_vals
        
        ### Spiral case
        elif geom_type == 'spiral' and attributes is not None:
            # Basic approximation for spirals - treat as a series of arcs with changing curvature
            # This is a simplified approach - for accurate spirals, use the Fresnel integrals
            curvStart = float(attributes.get('curvStart', 0))
            curvEnd = float(attributes.get('curvEnd', 0))
            
            # If both curvatures are 0, treat as a line
            if abs(curvStart) < 1e-10 and abs(curvEnd) < 1e-10:
                for s in s_array:
                    x = x0 + s * np.cos(hdg)
                    y = y0 + s * np.sin(hdg)
                    x_vals.append(x)
                    y_vals.append(y)
                return x_vals, y_vals
            
            # Approximate spiral as a series of arcs with gradually changing curvature
            current_x, current_y = x0, y0
            current_hdg = hdg
            
            for i in range(n_pts - 1):
                s_start = s_array[i]
                s_end = s_array[i+1]
                s_mid = (s_start + s_end) / 2
                segment_length = s_end - s_start
                
                # Calculate curvature at this point along the spiral
                t = s_mid / length  # Normalized position along spiral (0 to 1)
                current_curv = curvStart + t * (curvEnd - curvStart)
                
                if abs(current_curv) < 1e-10:
                    # Nearly straight segment
                    next_x = current_x + segment_length * np.cos(current_hdg)
                    next_y = current_y + segment_length * np.sin(current_hdg)
                else:
                    # Arc segment
                    radius = abs(1.0 / current_curv)
                    arc_direction = np.sign(current_curv)
                    dtheta = segment_length / radius * arc_direction
                    
                    # Update heading
                    next_hdg = current_hdg + dtheta
                    
                    # Calculate next point
                    next_x = current_x + segment_length * np.cos((current_hdg + next_hdg) / 2)
                    next_y = current_y + segment_length * np.sin((current_hdg + next_hdg) / 2)
                    
                    current_hdg = next_hdg
                
                x_vals.append(current_x)
                y_vals.append(current_y)
                
                current_x, current_y = next_x, next_y
            
            # Add the final point
            x_vals.append(current_x)
            y_vals.append(current_y)
            
            return x_vals, y_vals
        
        ### Poly3 case
        elif geom_type in ['poly3', 'paramPoly3'] and attributes is not None:
            log.error(f"Unsupported geometry type: {geom_type}. Will use default line approximation.")

        ### Line case (default)
        for s in s_array:
            x = x0 + s * np.cos(hdg)
            y = y0 + s * np.sin(hdg)
            x_vals.append(x)
            y_vals.append(y)
        return x_vals, y_vals
