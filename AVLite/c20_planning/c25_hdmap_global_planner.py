import networkx as nx
import logging

from c10_perception.c18_hdmap import HDMap
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c21_planning_model import GlobalPlan

log = logging.getLogger(__name__)

class HDMapGlobalPlanner(GlobalPlannerStrategy):
    """
    A global planner that uses OpenDRIVE HD maps for path planning.
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
        self.lane_path = [self.hdmap.lane_by_uid[lane_uid] for lane_uid in path]



    # TODO: The weights are road weight, not lanes. Change it later for precise distance calculation 
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


