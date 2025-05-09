import xml.etree.ElementTree as ET
from botocore.utils import conditionally_enable_crc32
from scipy.spatial import KDTree
import numpy as np
import networkx as nx
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

    def __init__(self, xodr_file:str, sampling_resolution=0.5):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution

        self.hdmap:HDMap = HDMap(xodr_file_name=xodr_file, sampling_resolution=sampling_resolution)
        self.hdmap.parse_HDMap()
        self.road_path = []
        
        log.debug(f"Loading HDMap from {xodr_file}")

    def plan(self):

        if not self.hdmap.road_network or not self.global_plan.start_point or not self.global_plan.goal_point:
            log.error("Road network or start/goal points not set.")
            return

        start_road = self.hdmap.find_nearest_road(*self.global_plan.start_point)
        goal_road = self.hdmap.find_nearest_road(*self.global_plan.goal_point)

        if not start_road or not goal_road:
            log.error("Start or goal road not found.")
            return


        # Plan global path
        path = self.__plan_global_path(start_road.id, goal_road.id)
        log.debug(f"Global path planned: {path}")
        self.road_path = [self.hdmap.road_ids[road_id] for road_id in path]


        
    def __generate_lane_graph(self):
        self.graph = nx.DiGraph()
        raise NotImplementedError("Lane graph generation not implemented yet.")

    def __plan_global_path(self, source_id:str, destination_id:str) -> list[str]:
        """
        Plan a path between two road IDs using Dijkstra or A* algorithm.
        """
        try:
            path = nx.shortest_path(self.hdmap.road_network, source=source_id, target=destination_id)
            log.debug(f"Path found from {source_id} to {destination_id}: {path}")
            return path
        except nx.NetworkXNoPath:
            log.error(f"No path found between {source_id} and {destination_id}.")
            return []


@dataclass
class HDMap:
    """Compact HD map representation for global planning"""
    
    @dataclass
    class Lane:
        id: str
        lane_element: ET.Element
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))      # Nx2 array of (x,y) coordinates
        left_d: list[float] = field(default_factory=list)         # Left boundary distances for each centerline point
        right_d: list[float] = field(default_factory=list)        # Right boundary distances for each centerline point
        road: Optional["HDMap.Road"] = None
        predecessor: Optional['HDMap.Lane'] = None
        successor: Optional['HDMap.Lane'] = None
        pred_id: str = "-1"  # ID of predecessor lane 
        succ_id: str = "-1"
        side : str = 'left'  # 'left' or 'right'
        type: str = 'driving'  # 'driving', 'shoulder', etc.
        road_id: str = "-1"  # ID of the road this lane belongs to

    @dataclass
    class Road:
        """Compact road representation for global planning"""
        id: str
        road_element:ET.Element
        pred_id: str = "-1"  
        succ_id: str = "-1"
        pred_type: str = "road"  # 'road' or 'junction'
        succ_type: str = "road"
        length: float = 0.0
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))      
        lanes: list['HDMap.Lane'] = field(default_factory=list)

    sampling_resolution: float = 1.0
    road_ids: dict[str, Road] = field(default_factory=dict)
    lane_ids: dict[str, Lane] = field(default_factory=dict)
    junction_ids: dict[str, list[Road]] = field(default_factory=dict)
    roads: list[Road] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list) 

    
    point_to_road: dict[tuple[int, int], Road] = field(default_factory=dict)
    point_to_lane: dict[tuple[int, int], Lane] = field(default_factory=dict)

    xodr_file_name: str = ""

    _all_road_points: list[tuple[float, float]] = field(default_factory=list)
    __all_lane_points: list[tuple[float, float]] = field(default_factory=list)
    __road_kdtree: Optional[KDTree] = None
    __lane_kdtree: Optional[KDTree] = None

    road_network: Optional[nx.DiGraph] = None
    

    def parse_HDMap(self) -> None:
        """Parse OpenDRIVE file and build lane graph"""
        if self.xodr_file_name == "":
            log.error("No OpenDRIVE file specified.")
            return

        try:
            tree = ET.parse(self.xodr_file_name)
            root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing OpenDRIVE file: {e}")
            return

        roads = root.findall('road')
        log.debug(f"Number of roads in HD Map: {len(roads)}")
        
        self.road_network = nx.DiGraph()
        
        # Store all road coordinates to calculate plot limits
        
        for road_element in roads:
            plan_view = road_element.find('planView')
            if plan_view is None:
                continue
                
            # Process road geometry to get centerline
            road_x, road_y = [], []
            
            # Extract all geometry segments first of the reference line
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
                
                # log.debug(f"n_pts: {int(length//self.sampling_resolution+1)}")
                x_vals, y_vals = self.sample_OpenDrive_geometry(x0, y0, hdg, length, gtype, attrib, n_pts=int(length//self.sampling_resolution+1))
                road_x.extend(x_vals)
                road_y.extend(y_vals)

            p_id, s_id = "-1", "-1"
            p_ids, s_ids = [], []
            predecessor = road_element.find("link/predecessor")
            if predecessor is not None:
                if predecessor.get('elementType') == 'road':
                    p_id = predecessor.get('elementId', '-1')
                elif predecessor.get('elementType') == 'junction':
                    # Extract all roads connected to the junction
                    junction_id = predecessor.get('elementId', '-1')
                    p_id = junction_id
                    p_ids = self.__get_road_successors_from_junction(root, road_element, junction_id)

            successor = road_element.find("link/successor")
            if successor is not None:
                if successor.get('elementType') == 'road':
                    s_id = successor.get('elementId', '-1')
                elif successor.get('elementType') == 'junction':
                    # Extract all roads connected to the junction
                    junction_id = successor.get('elementId', '-1')
                    s_id = junction_id
                    s_ids = self.__get_road_successors_from_junction(root,road_element, junction_id)

            r = HDMap.Road(
                id=road_element.get('id', '-1'),
                road_element=road_element,
                center_line=np.array([road_x, road_y]),
                pred_id=p_id,
                succ_id=s_id,
                length=float(road_element.get('length', '0')),
                pred_type=predecessor.get('elementType', 'road'),
                succ_type=successor.get('elementType', 'road'),
            )

            #########################
            # Creating networkx graph
            #########################
            self.road_network.add_node(r.id, road=r)
            # Add edges to the road networkx
            if r.pred_id != "-1" and r.pred_type == "road":
                self.road_network.add_edge(r.pred_id, r.id)
            if r.succ_id != "-1" and r.succ_type == "road":
                self.road_network.add_edge(r.id, r.succ_id)

            for succ in s_ids:
                self.road_network.add_edge(r.id, succ)
            for pred in p_ids:
                self.road_network.add_edge(pred, r.id)

            #########################
            for x, y in zip(x_vals, y_vals):
                self.point_to_road[(x, y)] = r
                self._all_road_points.append((x, y))

            self.road_ids[r.id] = r
            self.roads.append(r)
            self.__process_road_lanes(r, road_x, road_y)

        # Build KDTree for spatial queries
        self.__road_kdtree = KDTree(self._all_road_points)
        # self.__lane_kdtree = KDTree(self.__all_lane_points)
    
    def __get_road_successors_from_junction(self, root, road_element, junction_id):
        """
        Extract all successor road IDs connected to a junction for a given road.
        """
        junction = root.find(f".//junction[@id='{junction_id}']")
        if junction is None:
            log.warning(f"Junction with ID {junction_id} not found.")
            return []

        successor_roads = []
        road_id = road_element.get('id', '-1')

        for connection in junction.findall("connection"):
            incoming_road = connection.get("incomingRoad")
            if incoming_road == road_id:
                connecting_road = connection.get("connectingRoad")
                c_road_element = root.find(f".//road[@id='{connecting_road}']")
                if connecting_road and self.__road_has_driving_lanes(c_road_element):
                    successor_roads.append(connecting_road)

        log.debug(f"Junction {junction_id} connects road {road_id} to successors: {successor_roads}")
        return successor_roads


    def __road_has_driving_lanes(self, road_element) -> bool:
        """
        Check if a road element has at least one driving lane.
        """
        lane_sections = road_element.findall(".//laneSection")
        
        for section in lane_sections:
            lanes = section.findall(".//lane")
            for lane in lanes:
                lane_type = lane.get("type")
                if lane_type == "driving":
                    return True
        
        return False

    def __process_road_lanes(self, r:Road, road_x: list[float], road_y:list[float]):
        """Plot lanes for a given road"""

        road = r.road_element
        lanes_sections = road.findall('lanes/laneSection')
        if not lanes_sections:
            return
            
        lane_offsets = road.findall('lanes/laneOffset')
            
        for lane_section in lanes_sections:
            s_section = float(lane_section.get('s', '0.0'))
            offset = self.__get_lane_offset_at_s(lane_offsets, s_section)

            for side in ['left', 'right']:
                lanes = lane_section.findall(f"{side}/lane")
                if side == 'left':
                    lanes.sort(key=lambda l: int(l.get('id', '0')))  
                else:
                    lanes.sort(key=lambda l: int(l.get('id', '0')), reverse=True)  # Sort by decreasing lane ID

                cumulative_offset = offset  # Start with lane offset
                for lane in lanes:
                    lane_id = int(lane.get('id', '0'))
                    lane_type = lane.get('type', 'none')
                    width_element = lane.find('width')
                    
                    if width_element is not None and lane_id != 0:
                        width = float(width_element.get('a', '0'))
                        cumulative_offset += width/2 if side == 'left' else -width/2
                        
                        self.__set_lane(road_x, road_y, cumulative_offset, lane, r)
                        
                        cumulative_offset += width/2 if side == 'left' else -width/2

    

    def __set_lane(self, road_x, road_y, offset, lane_element, road: Road):
        """Plot a lane boundary at the specified offset from the road centerline"""

        assert len(road_x) == len(road_y), "Road X and Y coordinates must be of the same length."
        
        if not road_x or len(road_x) < 2:
            return
        
        # Calculate lane boundary points
        lane_x, lane_y = [], []
        
            
        for i in range(len(road_x)):
            try:
                # Find previous and next valid indices
                prev_idx = i - 1
                next_idx = i + 1
                
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
                log.error(f"Error processing lane boundary: {e}")
                continue
        
        if lane_x and lane_y:
            l = HDMap.Lane(
                id=lane_element.get('id', '0'),
                lane_element=lane_element,
                center_line=np.array([lane_x, lane_y]),
                side=lane_element.get('side', 'left'),
                type=lane_element.get('type', 'driving'),
                pred_id=lane_element.get('predecessor', '-1'),
                succ_id=lane_element.get('successor', '-1'),
                road_id = road.id,
                road = road,
            )
            self.lanes.append(l)
            for x, y in zip(lane_x, lane_y):
                self.point_to_lane[(x, y)] = l

    
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


    def find_nearest_road(self, x:float, y:float) -> Road|None:
        """Find the nearest road to the given coordinates"""
        if self.__road_kdtree is not None:
            _, index = self.__road_kdtree.query((x, y))
            if index >= 0 and index < len(self._all_road_points):
                x,y = self._all_road_points[index]
                return self.point_to_road.get((x, y), None)
        
    
    #TODO 
    def find_nearest_lane(self, position: tuple[float, float], k: int = 5) -> Lane:
        """Find lane closest to position"""
        pass

    #TODO
    def __connect_lanes(self, from_lane_id: str, to_lane_id: str) -> None:
        """Connect two lanes as predecessor/successor"""
        pass
    
    #TODO
    def __connect_roads(self, from_road_id: str, to_road_id: str) -> None:
        """Connect two roads as predecessor/successor"""
        pass
    
    




    def sample_OpenDrive_geometry(self, x0, y0, hdg, length, geom_type='line', attributes=None, n_pts=50):
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
