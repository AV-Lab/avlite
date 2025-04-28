import xml.etree.ElementTree as ET
from scipy.spatial import KDTree
import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline
from typing import Optional
import logging
from dataclasses import dataclass, field

from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c21_planning_model import GlobalPlan

log = logging.getLogger(__name__)

class HDMapGlobalPlanner(GlobalPlannerStrategy):
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

        log.debug(f"Loading HDMap from {xodr_file}")
        self.hdmap = HDMap(xodr_file_name=xodr_file)
        self.hdmap.parse_HDMap()


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


@dataclass
class HDMap:
    """Compact HD map representation for global planning"""
    
    @dataclass
    class Lane:
        id: str
        center_line: np.ndarray      # Nx2 array of (x,y) coordinates
        left_d: list[float]          # Left boundary distances for each centerline point
        right_d: list[float]         # Right boundary distances for each centerline point
        road: Optional["HDMap.Road"] = None
        predecessors: list['HDMap.Lane'] = field(default_factory=list)
        successors: list['HDMap.Lane'] = field(default_factory=list)
        side : str = 'left'  # 'left' or 'right'
        type: str = 'driving'  # 'driving', 'shoulder', etc.

    @dataclass
    class Road:
        id: str
        center_line: np.ndarray      # Nx2 array of (x,y) coordinates
        lanes: list['HDMap.Lane'] = field(default_factory=list)

    road_predecessors: dict[Road, list[Road]] = field(default_factory=dict)
    road_successors:  dict[Road, list[Road]] = field(default_factory=dict)
    lane_predecessors: dict[Lane, list[Lane]] = field(default_factory=dict)
    lane_successors:  dict[Lane, list[Lane]] = field(default_factory=dict)
    xodr_file_name: str = ""

    roads: list[Road] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list) # Key is lane type driving, shoulder, etc.

    _kdtree: Optional[KDTree] = None
    _point_to_lane: list[tuple[str, int]] = field(default_factory=list)
    
    #TODO
    def __connect_lanes(self, from_lane_id: str, to_lane_id: str) -> None:
        """Connect two lanes as predecessor/successor"""
        pass
    
    #TODO
    def __connect_roads(self, from_road_id: str, to_road_id: str) -> None:
        """Connect two roads as predecessor/successor"""
        pass
    
    #TODO
    def __build_spatial_index(self) -> None:
        """Build KDTree for efficient position queries"""
        points = []
        self._point_to_lane = []
        
        for lane_id, lane in self.lanes.items():
            for i, point in enumerate(lane.center_line):
                points.append(point)
                self._point_to_lane.append((lane_id, i))
                
        self._kdtree = KDTree(np.array(points))
    
    #TODO 
    def find_nearest_lane(self, position: tuple[float, float], k: int = 5) -> Lane:
        """Find lane closest to position"""
        if self._kdtree is None:
            self.__build_spatial_index()
            
        _, indices = self._kdtree.query(position, k=k)
        nearby_lanes = {self._point_to_lane[i][0] for i in indices}
        
        return min(
            [self.lanes[lane_id] for lane_id in nearby_lanes],
            key=lambda l: np.min(np.linalg.norm(l.center_line - np.array(position), axis=1))
        )

    def parse_HDMap(self) -> None:
        """Parse OpenDRIVE file and build lane graph"""
        if self.xodr_file_name == "":
            log.error("No OpenDRIVE file specified.")
            return

        try:
            tree = ET.parse(self.xodr_file_name)
            self.xodr_root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing OpenDRIVE file: {e}")

        root = self.xodr_root
        roads = root.findall('road')
        log.debug(f"Number of roads in HD Map: {len(roads)}")
        
        # Store all road coordinates to calculate plot limits
        
        for road in roads:
            plan_view = road.find('planView')
            if plan_view is None:
                continue
                
            # Process road geometry to get centerline
            road_x, road_y = [], []
            
            # Extract all geometry segments first
            for geometry in plan_view.findall('geometry'):
                x0 = float(geometry.get('x', '0'))
                y0 = float(geometry.get('y', '0'))
                hdg = float(geometry.get('hdg', '0'))
                length = float(geometry.get('length', '0'))
                gtype = 'line'  # Default to line if no specific geometry type is found
                attrib = {}
                
                # Check for all possible geometry types in OpenDRIVE
                for child in geometry:
                    if child.tag in ['line', 'arc', 'spiral', 'poly3', 'paramPoly3']:
                        gtype = child.tag
                        attrib = child.attrib
                        break
                
                x_vals, y_vals = sample_OpenDrive_geometry(x0, y0, hdg, length, gtype, attrib)
                # if road_x:  # add gap between consecutive segments
                #     road_x.append(np.nan)
                #     road_y.append(np.nan)
                road_x.extend(x_vals)
                road_y.extend(y_vals)
                r = HDMap.Road(
                    id=road.get('id', '0'),
                    center_line=np.array([road_x, road_y]),
                )
                self.roads.append(r)
                
            
            self.__set_road_lanes(road, road_x, road_y)

    def __set_road_lanes(self, road, road_x, road_y):
        """Plot lanes for a given road"""
        lanes_sections = road.findall('lanes/laneSection')
        if not lanes_sections:
            return
            
        # Get lane offsets if they exist
        lane_offsets = road.findall('lanes/laneOffset')
        road_id = road.get('id', '0')
            
        for lane_section in lanes_sections:
            # Process left lanes (positive IDs)
            left_lanes = lane_section.findall('left/lane')
            left_lanes.sort(key=lambda l: int(l.get('id', '0')))  # Sort by increasing lane ID
            
            # Apply lane offset at the section s-coordinate
            s_section = float(lane_section.get('s', '0.0'))
            offset = self.__get_lane_offset_at_s(lane_offsets, s_section)
            
            cumulative_offset = offset  # Start with lane offset
            for lane in left_lanes:
                lane_id = int(lane.get('id', '0'))
                lane_type = lane.get('type', 'none')
                width_element = lane.find('width')
                
                if width_element is not None and lane_id > 0:
                    width = float(width_element.get('a', '0'))
                    cumulative_offset += width
                    # if lane_type == "driving":
                    self.__set_lane_boundary(road_x, road_y, cumulative_offset, lane_type, lane_id,road, 'left')
            
            # Process right lanes (negative IDs)
            right_lanes = lane_section.findall('right/lane')
            right_lanes.sort(key=lambda l: int(l.get('id', '0')), reverse=True)  # Sort by decreasing lane ID
            
            cumulative_offset = offset  # Reset with base lane offset for right lanes
            for lane in right_lanes:
                lane_id = int(lane.get('id', '0'))
                lane_type = lane.get('type', 'none')
                width_element = lane.find('width')
                
                if width_element is not None and lane_id < 0:
                    width = float(width_element.get('a', '0'))
                    cumulative_offset -= width  # Negative because it's on the right side
                    # if lane_type == "driving":
                    self.__set_lane_boundary(road_x, road_y, cumulative_offset, lane_type,lane_id, road,'right')
    
    def __get_lane_offset_at_s(self, lane_offsets, s):
        """Calculate lane offset at position s using the OpenDRIVE lane offset elements."""
        if not lane_offsets:
            return 0.0
            
        # Find the applicable lane offset element
        applicable_offset = None
        for offset in lane_offsets:
            offset_s = float(offset.get('s', '0.0'))
            if offset_s <= s:
                applicable_offset = offset
            else:
                break
                
        if applicable_offset is None:
            return 0.0
            
        # Calculate offset using polynomial
        offset_s = float(applicable_offset.get('s', '0.0'))
        local_s = s - offset_s
        a = float(applicable_offset.get('a', '0.0'))
        b = float(applicable_offset.get('b', '0.0'))
        c = float(applicable_offset.get('c', '0.0'))
        d = float(applicable_offset.get('d', '0.0'))
        
        return a + b*local_s + c*local_s**2 + d*local_s**3

    def __set_lane_boundary(self, road_x, road_y, offset, lane_type, lane_id, road, side):
        """Plot a lane boundary at the specified offset from the road centerline"""
        if not road_x or len(road_x) < 2:
            return
        
        # Calculate lane boundary points
        lane_x, lane_y = [], []
        valid_indices = [i for i, x in enumerate(road_x) if not np.isnan(x)]
        
        if not valid_indices:
            return
            
        for i in valid_indices:
            try:
                # Find previous and next valid indices
                prev_idx = i - 1
                while prev_idx >= 0 and np.isnan(road_x[prev_idx]):
                    prev_idx -= 1
                    
                next_idx = i + 1
                while next_idx < len(road_x) and np.isnan(road_x[next_idx]):
                    next_idx += 1
                
                # Calculate direction vector
                if prev_idx >= 0 and next_idx < len(road_x):
                    # Use both previous and next points for smoother transitions
                    dx1 = road_x[i] - road_x[prev_idx]
                    dy1 = road_y[i] - road_y[prev_idx]
                    dx2 = road_x[next_idx] - road_x[i]
                    dy2 = road_y[next_idx] - road_y[i]
                    dx = (dx1 + dx2) / 2
                    dy = (dy1 + dy2) / 2
                elif prev_idx >= 0:
                    # Only previous point available
                    dx = road_x[i] - road_x[prev_idx]
                    dy = road_y[i] - road_y[prev_idx]
                elif next_idx < len(road_x):
                    # Only next point available
                    dx = road_x[next_idx] - road_x[i]
                    dy = road_y[next_idx] - road_y[i]
                else:
                    continue
                
                # Calculate unit vector normal to the road direction
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Normal vector pointing outward from the road
                    nx = -dy / length
                    ny = dx / length
                    
                    # Add offset point to the lane boundary
                    lane_x.append(road_x[i] + nx * offset)
                    lane_y.append(road_y[i] + ny * offset)
                
            except (IndexError, ValueError) as e:
                continue
        
        if lane_x and lane_y:
            self.lanes.append(HDMap.Lane(
                id=lane_id,
                center_line=np.array([lane_x, lane_y]),
                left_d=[0]*len(lane_x),
                right_d=[0]*len(lane_x),
                road=road,
                type=lane_type,
            ))

            # self.ax.plot(lane_x, lane_y, color=color, alpha=alpha, linewidth=1.5)



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
