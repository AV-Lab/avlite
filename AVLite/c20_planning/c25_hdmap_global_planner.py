from sys import exc_info
import xml.etree.ElementTree as ET
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
    A global planner that uses OpenDRIVE HD maps for path planning.
    """

    def __init__(self, xodr_file:str, sampling_resolution=1):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution

        self.hdmap:HDMap = HDMap(xodr_file_name=xodr_file, sampling_resolution=sampling_resolution)
        self.path = []
        self.lane_path = []
        
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


        start_lane = self.hdmap.find_nearest_lane(*self.global_plan.start_point)
        goal_lane = self.hdmap.find_nearest_lane(*self.global_plan.goal_point)
        log.info(f"Start lane: {start_lane.uid}, Goal lane: {goal_lane.uid}")

        if not start_lane or not goal_lane:
            log.error("Start or goal lane not found.")
            return


        # Plan global path
        # path = self.__plan_global_path(start_road.id, goal_road.id)
        # self.road_path = [self.hdmap.road_ids[road_id] for road_id in path]
        path = self.__plan_global_path(start_lane.uid, goal_lane.uid)
        self.road_path = [self.hdmap.lane_uids[lane_uid] for lane_uid in path]
        log.debug(f"Global path planned: {path}")



    # TODO:  
    def __plan_global_path(self, source_id:str, destination_id:str) -> list[str]:
        """
        Plan a path between two road IDs using Dijkstra or A* algorithm.
        """
        try:
            # Debug: print node info and types
            log.debug(f"All lane_network nodes: {list(self.hdmap.lane_network.nodes())[:20]} ...")
            log.debug(f"Source_id: {repr(source_id)}, type: {type(source_id)}")
            log.debug(f"Destination_id: {repr(destination_id)}, type: {type(destination_id)}")
            if source_id not in self.hdmap.lane_network or destination_id not in self.hdmap.lane_network:
                log.error(f"Source {source_id} or destination {destination_id} not in lane network.")
                return []
            path = nx.shortest_path(self.hdmap.lane_network, source=source_id, target=destination_id)
            log.debug(f"Path found from {source_id} to {destination_id}: {path}")
            return path
        except Exception as e: 
            log.error(f"No path found between {source_id} and {destination_id}. Error: {e}")
            return []


@dataclass
class HDMap:
    """Compact HD map representation for global planning"""
    
    @dataclass
    class Lane:
        id: str
        uid: str # Unique ID for the lane 
        lane_element: ET.Element
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))      # 2xN array of (x,y) coordinates
        left_d: list[float] = field(default_factory=list)         # Left boundary distances for each centerline point
        right_d: list[float] = field(default_factory=list)        # Right boundary distances for each centerline point
        road: Optional["HDMap.Road"] = None
        pred_id: str = ""  # ID of predecessor lane 
        succ_id: str = ""
        pred_type: str = "lane"  # 'lane' or 'junction'
        succ_type: str = "lane"  # 'lane' or 'junction'
        side : str = 'left'  # 'left' or 'right'
        type: str = 'driving'  # 'driving', 'shoulder', etc.
        road_id: str = ""  # ID of the road this lane belongs to
        width: float = 0.0  
        lane_section_idx: int = 0  # Lane section s 
        predecessors: list['HDMap.Lane'] = field(default_factory=list)
        successors: list['HDMap.Lane'] = field(default_factory=list)

    @dataclass
    class Road:
        """Compact road representation for global planning"""
        id: str
        road_element:ET.Element
        pred_id: str = ""  
        succ_id: str = ""
        pred_type: str = "road"  # 'road' or 'junction'
        succ_type: str = "road"
        length: float = 0.0
        junction_id: str = ""  
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))      
        predecessors: list['HDMap.Road'] = field(default_factory=list)
        successors: list['HDMap.Road'] = field(default_factory=list)
        lane_sections: list[list['HDMap.Lane']] = field(default_factory=list) 
        lane_section_s_vals: list[float] = field(default_factory=list)  # List of lane section s values

    xodr_file_name: str 
    sampling_resolution: float = 1.0
    roads: list[Road] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list) 
    
    road_ids: dict[str, Road] = field(default_factory=dict)
    lane_uids: dict[str, Lane] = field(default_factory=dict)
    junction_ids: dict[str, list[Road]] = field(default_factory=dict)
    road_network: nx.DiGraph = field(default_factory=nx.DiGraph)
    lane_network: nx.DiGraph = field(default_factory=nx.DiGraph)

    __point_to_road: dict[tuple[int, int], Road] = field(default_factory=dict)
    __point_to_drivable_lane: dict[tuple[int, int], Lane] = field(default_factory=dict)
    __road_kdtree: Optional[KDTree] = None
    __lane_kdtree_drivable: Optional[KDTree] = None

    __all_road_points: list[tuple[float, float]] = field(default_factory=list)
    __all_drivable_lane_points: list[tuple[float, float]] = field(default_factory=list)


    def __post_init__(self):
        if self.xodr_file_name == "":
            log.error("No OpenDRIVE file specified.")
            return
        
        self.root = self.parse_HDMap()
        self.__road_kdtree = KDTree(self.__all_road_points)
        # self.__lane_kdtree = KDTree(self.__all_lane_points)
        self.__lane_kdtree_drivable = KDTree(self.__all_drivable_lane_points)
        self.__connect_roads()
        self.__connect_lanes()

        log.debug(f"Number of roads in HD Map: {len(self.roads)} vs nodes in graph {len(self.road_network.nodes())}")
        log.debug(f"Number of lanes in HD Map: {len(self.lanes)} vs nodes in graph {len(self.lane_network.nodes())}")
    

    def parse_HDMap(self):
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
            predecessor = road_element.find("link/predecessor")
            if predecessor is not None:
                if predecessor.get('elementType') == 'road':
                    p_id = predecessor.get('elementId', '')
                elif predecessor.get('elementType') == 'junction':
                    # Extract all roads connected to the junction
                    junction_id = predecessor.get('elementId', '')
                    p_id = junction_id

            successor = road_element.find("link/successor")
            if successor is not None:
                if successor.get('elementType') == 'road':
                    s_id = successor.get('elementId', '')
                elif successor.get('elementType') == 'junction':
                    # Extract all roads connected to the junction
                    junction_id = successor.get('elementId', '')
                    s_id = junction_id

            r = HDMap.Road(
                id=road_element.get('id', ''),
                road_element=road_element,
                center_line=np.array([road_x, road_y]),
                pred_id=p_id,
                succ_id=s_id,
                length=float(road_element.get('length', '')),
                pred_type=predecessor.get('elementType', '') if predecessor is not None else '',
                succ_type=successor.get('elementType', '') if successor is not None else '',
                junction_id=road_element.get('junction', ''),
            )


            for x, y in zip(road_x, road_y):
                self.__point_to_road[(x, y)] = r
                self.__all_road_points.append((x, y))

            if self.road_ids.get(r.id) is not None:
                log.error(f"Adding Road ID {r.id}, but it already exists in road_ids.")
            self.road_ids[r.id] = r
            self.roads.append(r)
            self.__process_lane_sections(r, road_x, road_y)

        return root

    


    def __process_lane_sections(self, r:Road, road_x: list[float], road_y:list[float]):
        """Plot lanes for a given road"""

        road = r.road_element
        lanes_sections = road.findall('lanes/laneSection')
        if not lanes_sections:
            return
            
        lane_offsets = road.findall('lanes/laneOffset')
            
        for i,lane_section in enumerate(lanes_sections):
            r.lane_sections.append([])
            s_section = float(lane_section.get('s', '0.0'))
            r.lane_section_s_vals.append(s_section)
            offset = self.__get_lane_offset_at_s(lane_offsets, s_section)

            for side in ['left', 'right']:
                lanes = lane_section.findall(f"{side}/lane")
                if side == 'left':
                    lanes.sort(key=lambda l: int(l.get('id', '0')))  
                else:
                    lanes.sort(key=lambda l: int(l.get('id', '0')), reverse=True)  # Sort by decreasing lane ID

                cumulative_offset = offset  # Start with lane offset
                for lane_element in lanes:
                    lane_id = int(lane_element.get('id', '0'))
                    width_element = lane_element.find('width')
                    
                    if width_element is not None and lane_id != 0:
                        width = float(width_element.get('a', '0.0'))
                        if float(width_element.get('b')) != 0.0 and lane_element.get('type') == 'driving':
                            log.warning(f"Lanes with variable width are not supported yet. We assume fixed width of \
                                   {width:.2f} Lane ID: {lane_id}, type: {lane_element.get('type')}, Road ID: {r.id}")

                        cumulative_offset += width/2 if side == 'left' else -width/2
                        
                        #########
                        pred_id = lane_element.find('link/predecessor').get('id', '') if lane_element.find('link/predecessor') is not None else ''
                        succ_id = lane_element.find('link/successor').get('id', '') if lane_element.find('link/successor') is not None else ''
                        l = HDMap.Lane(
                            id=lane_element.get('id', ''),
                            uid=f"{r.id}_{lane_element.get('id', '')}",
                            lane_element=lane_element,
                            side=lane_element.get('side', ''),
                            type=lane_element.get('type', ''),
                            pred_id=pred_id,
                            succ_id=succ_id,
                            road_id = r.id,
                            lane_section_idx = i,
                            road = r,
                        )
                        self.lanes.append(l)
                        r.lane_sections[i].append(l)


                        lane_x, lane_y = self.__get_lane_xy(road_x, road_y, cumulative_offset, l=l, road=r)
                        if lane_x and lane_y:
                            l.center_line = np.array([lane_x, lane_y])
                            for x, y in zip(lane_x, lane_y):
                                x = float(x)
                                y = float(y)
                                # self.__point_to_lane[(x, y)] = l
                                # self.__all_lane_points.append((x, y))
                                if l.type == 'driving':
                                    self.__point_to_drivable_lane[(x, y)] = l
                                    self.__all_drivable_lane_points.append((x, y))

                        #########

                        cumulative_offset += width/2 if side == 'left' else -width/2

    

    def __get_lane_xy(self, road_x, road_y, offset, l:Lane, road: Road) -> tuple[list[float], list[float]]:
        """Plot a lane boundary at the specified offset from the road centerline"""

        assert len(road_x) == len(road_y), "Road X and Y coordinates must be of the same length."
        
        if not road_x or len(road_x) < 2:
            return
        
        lane_x, lane_y = [], [] # center line points of the lane
            
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
                    # Weight the average to reduce sharp transitions
                    dx = (dx1 + dx2) / 2
                    dy = (dy1 + dy2) / 2
                elif prev_idx >= 0:
                    # Last point - use previous two points if available
                    if prev_idx > 0:
                        dx1 = road_x[prev_idx] - road_x[prev_idx-1]
                        dy1 = road_y[prev_idx] - road_y[prev_idx-1]
                        dx2 = road_x[i] - road_x[prev_idx]
                        dy2 = road_y[i] - road_y[prev_idx]
                        dx = (dx1 + dx2) / 2
                        dy = (dy1 + dy2) / 2
                    else:
                        # Only previous point available
                        dx = road_x[i] - road_x[prev_idx]
                        dy = road_y[i] - road_y[prev_idx]
                elif next_idx < len(road_x):
                    # First point - use next two points if available
                    if next_idx + 1 < len(road_x):
                        dx1 = road_x[next_idx] - road_x[i]
                        dy1 = road_y[next_idx] - road_y[i]
                        dx2 = road_x[next_idx+1] - road_x[next_idx]
                        dy2 = road_y[next_idx+1] - road_y[next_idx]
                        dx = (dx1 + dx2) / 2
                        dy = (dy1 + dy2) / 2
                    else:
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

        return lane_x, lane_y
        

    
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

    def lane_reversed(self, lane:Lane) -> bool:
        """
        Determine if lane is reversed by checking if when moving to successor,
        there are no lanes on the left-hand side.
        """
        for successor in lane.successors:
            succ_brothers = successor.road.lane_sections[successor.lane_section_idx]
            if successor.road and succ_brothers:
                if not any(lane.side == 'left' for lane in succ_brothers):
                    return True
        
        return False
    def _get_connecting_roads_from_junction(self, root, road_element, junction_id):
        """
        Extract all successor road IDs connected to a junction for a given road.
        """
        junction = root.find(f".//junction[@id='{junction_id}']")
        if junction is None:
            log.error(f"Junction with ID {junction_id} not found. road_id: {road_element.get('id')}")
            return []

        successor_roads = []
        road_id = road_element.get('id', '')

        for connection in junction.findall("connection"):
            incoming_road = connection.get("incomingRoad")
            if incoming_road == road_id:
                connecting_road = connection.get("connectingRoad")
                if connecting_road: # and self.__road_has_driving_lanes(c_road_element):
                    successor_roads.append(connecting_road)

        return successor_roads
    
    def __connect_lanes(self) -> None:
        """Connect all lanes based on their links and junction definitions, only adding reachable lanes to the graph."""
        lane_by_uid = {f"{l.road_id}_{l.id}": l for l in self.lanes if l.type == "driving"}
        self.lane_uids = lane_by_uid

        # Build connections
        for lane in self.lanes:
            if lane.type != "driving":
                continue

            lane_uid = f"{lane.road_id}_{lane.id}"
            # TODO: Remove this 
            # self.lane_network.add_node(lane_uid)

            # Predecessors
            if lane.pred_id and lane.pred_type == "lane":
                for pred_road in lane.road.predecessors:
                    pred_uid = f"{pred_road.id}_{lane.pred_id}"
                    pred_lane = lane_by_uid.get(pred_uid)
                    if pred_lane and lane not in pred_lane.successors:
                        lane.predecessors.append(pred_lane)
                        is_reversed = int(lane.id) * int(pred_lane.id) < 0
                        if is_reversed:
                            self.lane_network.add_edge(lane_uid, pred_uid, weight=lane.road.length)
                            if lane not in pred_lane.predecessors:
                                pred_lane.predecessors.append(lane)
                        else:
                            self.lane_network.add_edge(pred_uid, lane_uid, weight=pred_lane.road.length)
                            if lane not in pred_lane.successors:
                                pred_lane.successors.append(lane)
            # Successors
            if lane.succ_id and lane.succ_type == "lane":
                for succ_road in lane.road.successors:
                    succ_uid = f"{succ_road.id}_{lane.succ_id}"
                    succ_lane = lane_by_uid.get(succ_uid)
                    if succ_lane and succ_lane not in lane.successors:
                        lane.successors.append(succ_lane)
                        is_reversed = int(lane.id) * int(pred_lane.id) < 0
                        if is_reversed:
                            self.lane_network.add_edge(succ_uid, lane_uid, weight=succ_lane.road.length)
                            if lane not in succ_lane.successors:
                                succ_lane.successors.append(lane)
                        else:
                            if lane not in succ_lane.predecessors:
                                succ_lane.predecessors.append(lane)
                            self.lane_network.add_edge(lane_uid, succ_uid, weight=lane.road.length)

        # Junction connections
        for junction in self.root.findall(".//junction"):
            junction_id = junction.get("id")
            for connection in junction.findall("connection"):
                incoming_road_id = connection.get("incomingRoad")
                connecting_road_id = connection.get("connectingRoad")
                for lane_link in connection.findall("laneLink"):
                    from_id = lane_link.get("from")
                    to_id = lane_link.get("to")
                    from_uid = f"{incoming_road_id}_{from_id}"
                    to_uid = f"{connecting_road_id}_{to_id}"
                    from_lane = lane_by_uid.get(from_uid)
                    to_lane = lane_by_uid.get(to_uid)
                    if from_lane and to_lane and to_lane not in from_lane.successors:
                        is_reversed = int(from_lane.id) * int(to_lane.id) < 0
                        if is_reversed:
                            self.lane_network.add_edge(to_uid, from_uid, weight=to_lane.road.length)
                            if from_lane not in to_lane.predecessors:
                                from_lane.predecessors.append(to_lane)
                            if to_lane not in from_lane.successors:
                                to_lane.successors.append(from_lane)
                        else:
                            self.lane_network.add_edge(from_uid, to_uid, weight=from_lane.road.length)
                            if to_lane not in from_lane.successors:
                                to_lane.successors.append(from_lane)
                            if from_lane not in to_lane.predecessors:
                                from_lane.predecessors.append(to_lane)

        # Remove isolated nodes (not connected to any other lane)
        # isolated = [n for n in self.lane_network.nodes if self.lane_network.degree(n) == 0]
        # self.lane_network.remove_nodes_from(isolated)

        log.debug(f"Lane network: {len(self.lane_network.nodes())} nodes, {len(self.lane_network.edges())} edges")

        # add edges between lanes of the same sign for each road
        for road in self.roads:
            for lane_section in road.lane_sections:
                for lane in lane_section:
                    if lane.type == "driving":
                        for other_lane in lane_section:
                            if other_lane.type == "driving" and lane != other_lane and int(lane.id) * int(other_lane.id) > 0:
                                # Add edges between lanes of the same sign
                                self.lane_network.add_edge(lane.uid, other_lane.uid, weight=0.0)
                                self.lane_network.add_edge(other_lane.uid, lane.uid, weight=0.0)


    
    def __connect_roads(self) -> None:
        """Connect two roads as predecessor/successor"""
        for r in self.roads:
            # Connect predecessor and successor roads
            if r.pred_id != "" and r.pred_type == "road":
                pred_road = self.road_ids.get(r.pred_id, None)
                if pred_road:
                    r.predecessors.append(pred_road)
                    # self.road_network.add_edge(r.pred_id, r.id, weight= pred_road.length)

            if r.succ_id != "" and r.succ_type == "road":
                succ_road = self.road_ids.get(r.succ_id, None)
                if succ_road:
                    r.successors.append(succ_road)
                    self.road_network.add_edge(r.id, r.succ_id, weight= r.length)
            
            #########################
            # Dealing with junctions
            #########################
            
            if self.road_has_driving_lanes(r):
                p_ids, s_ids = [], []
                successor = r.road_element.find("link/successor")
                if successor is not None and successor.get('elementType') == 'junction':
                    junction_id = successor.get('elementId', '')
                    s_ids = self._get_connecting_roads_from_junction(self.root, r.road_element, junction_id)

                predecessor = r.road_element.find("link/predecessor")
                if predecessor is not None and predecessor.get('elementType') == 'junction':
                    junction_id = predecessor.get('elementId', '')
                    p_ids = self._get_connecting_roads_from_junction(self.root, r.road_element, junction_id)

                # adding juction roads
                for succ in s_ids:
                    s_road = self.road_ids.get(succ, None)
                    assert s_road is not None, f"Road ID {succ} not found in road_ids."

                    if self.road_has_driving_lanes(s_road):
                        self.road_network.add_edge(r.id, succ, weight= r.length)
                        r.successors.append(self.road_ids[succ])


                for pred in p_ids:
                    p_road = self.road_ids.get(pred, None)
                    assert p_road is not None, f"Road ID {pred} not found in road_ids."


                    if self.road_has_driving_lanes(p_road):
                        self.road_network.add_edge(pred, r.id, weight= r.length)
                        r.predecessors.append(self.road_ids[pred])
            #########################
            #########################
    
    




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
    


    def find_nearest_road(self, x:float, y:float) -> Road|None:
        """Find the nearest road to the given coordinates"""
        if self.__road_kdtree is not None:
            _, index = self.__road_kdtree.query((x, y))
            if index >= 0 and index < len(self.__all_road_points):
                x,y = self.__all_road_points[index]
                p = self.__point_to_road.get((x, y), None)
                if p is None:
                    log.error(f"Point not found in point_to_road mapping: {(x, y)}")

                return self.__point_to_road.get((x, y), None)
        
    
    def find_nearest_lane(self, x:float, y:float) -> Lane|None:
        """Find lane closest to position"""
        if self.__lane_kdtree_drivable is not None:
            _, index = self.__lane_kdtree_drivable.query((x, y))
            if 0 <= index < len(self.__all_drivable_lane_points):
                x,y = self.__all_drivable_lane_points[index]
                p = self.__point_to_drivable_lane.get((x, y), None)
                if p is None:
                    log.error(f"Point not found in point_to_lane mapping: {(x, y)}")

                return self.__point_to_drivable_lane.get((x, y), None)
    
    def road_has_driving_lanes(self, road: Road) -> bool:
        """
        Check if a road element has at least one driving lane.
        """
        road_element = road.road_element
        lane_sections = road_element.findall(".//laneSection")
        
        for section in lane_sections:
            lanes = section.findall(".//lane")
            for lane in lanes:
                lane_type = lane.get("type")
                if lane_type == "driving":
                    return True
        return False
    def  road_is_bidirectional(self, road: Road) -> bool:
        """
        Check if a road element is bidirectional.
        """
        road_element = road.road_element
        lane_sections = road_element.findall(".//laneSection")
        
        right = False
        left = False
        for section in lane_sections:
            lanes = section.findall(".//lane")
            for lane in lanes:
                lane_type = lane.get("type")
                if lane_type == "driving" and int(lane.get("id", "0")) < 0:
                    right = True
                if lane_type == "driving" and int(lane.get("id", "0")) > 0:
                    left = True
                if right and left:
                    return True
        return False
