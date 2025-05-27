import networkx as nx
import numpy as np
import logging

from c10_perception.c18_hdmap import HDMap
from c20_planning.c21_planning_model import GlobalPlan
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c28_trajectory import Trajectory

log = logging.getLogger(__name__)

class HDMapGlobalPlanner(GlobalPlannerStrategy):
    """
    A global planner that uses OpenDRIVE HD maps for path planning.
    """

    def __init__(self, xodr_file:str, sampling_resolution=0.5, velocity=30):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution

        self.hdmap:HDMap = HDMap(xodr_file_name=xodr_file, sampling_resolution=sampling_resolution)
        self.velocity = velocity
        
        log.debug(f"Loading HDMap from {xodr_file}")

    def plan(self) -> GlobalPlan: 

        if not self.hdmap.road_network or not self.start_point or not self.goal_point:
            log.error("Road network or start/goal points not set.")
            return

        start_road = self.hdmap.find_nearest_road(*self.start_point)
        goal_road = self.hdmap.find_nearest_road(*self.goal_point)
        if not start_road or not goal_road:
            log.error("Start or goal road not found.")
            return


        start_lane, s_idx = self.hdmap.find_nearest_lane_and_idx(*self.start_point)
        goal_lane, g_idx = self.hdmap.find_nearest_lane_and_idx(*self.goal_point)
        log.info(f"Start lane: {start_lane.uid}, Goal lane: {goal_lane.uid}")

        if not start_lane or not goal_lane:
            log.error("Start or goal lane not found.")
            return


        # Plan global path
        path = self.__plan_global_path(start_lane.uid, goal_lane.uid)
        # removing lane changes by keeping only the final lane in that road. This will utilize lane_change attribute in the edge
        # Notice that nodes and lanes, and edges are the changes
        # TODO: Lane sections are not handled properly
        path2 = [path[0]]
        for i in range(1,len(path)):
            lane = self.hdmap.lane_by_uid[path[i]]
            if lane.road_id != self.hdmap.lane_by_uid[path2[-1]].road_id:
                path2.append(path[i])
            elif lane.lane_section_idx != self.hdmap.lane_by_uid[path2[-1]].lane_section_idx:
                path2.append(path[i])
                log.warning(f"Lane section change detected: {path2[-1]} to {path[i]}; however, this is not handled properly yet.")


        log.debug(f"Filtered Path:  {start_lane.uid} to {goal_lane.uid}: {path2}")

       
        self.global_plan = GlobalPlan()

        # Populating global_plan path
        # TODO: handle points that are added backwards
        for i, uid in enumerate(path2):
            lane = self.hdmap.lane_by_uid[uid]
            # if i == 0:
            #     log.debug(f"Start lane: {lane.uid}, point IDX: {s_idx}")
            #     for point in self.__chop_path(lane, s_idx, start=True):
            #         self.global_plan.path.append(point)
            #         self.global_plan.left_boundary_d.append(lane.width/2)
            #         self.global_plan.right_boundary_d.append(-lane.width/2)
            #         self.global_plan.velocity.append(self.velocity)
            # elif i == len(path2) - 1:
            #     log.debug(f"Goal lane: {lane.uid}, point IDX: {g_idx}")
            #     for point in self.__chop_path(lane, g_idx, start=False):
            #         self.global_plan.path.append(point)
            #         self.global_plan.left_boundary_d.append(lane.width/2)
            #         self.global_plan.right_boundary_d.append(-lane.width/2)
            #         self.global_plan.velocity.append(self.velocity)
            # else:
            for point in zip(lane.center_line[0], lane.center_line[1]):
                self.global_plan.path.append(point)
                self.global_plan.left_boundary_d.append(lane.width/2)
                self.global_plan.right_boundary_d.append(-lane.width/2)
                self.global_plan.velocity.append(self.velocity)

        self.global_plan.lane_path = [self.hdmap.lane_by_uid[lane_uid] for lane_uid in path2]
        # self.global_plan.trajectory = Trajectory(path=path2, velocity=self.global_plan.velocity)

        return self.global_plan

    # TODO: The weights are road weight, not lanes. Change it later for precise distance calculation 
    def __plan_global_path(self, source_id:str, destination_id:str) -> list[str]:
        """
        Plan a path between two road IDs using Dijkstra or A* algorithm.
        """
        try:
            # Debug: print node info and types
            # log.debug(f"All lane_network nodes: {list(self.hdmap.lane_network.nodes())[:20]} ...")
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

def __chop_path(lane, idx, start=True):
    """
    Chops the path at a given index, either from the start or end.
    """

    path = lane.center_line.T
    # find if the path is getting closer to the idx or farther away
    idx_point = path[idx]
    s_point = np.array(path[0])
    sn_point = np.array(path[1])
    e_point = np.array(path[-1])
    en_point = np.array(path[-2])

    getting_closer_from_start = np.linalg.norm(s_point - idx_point) > np.linalg.norm(sn_point - idx_point)
    getting_closer_backward_from_end = np.linalg.norm(e_point - idx_point) > np.linalg.norm(en_point - idx_point)
            
    if start and lane.side=="right":
        return path[idx:] if getting_closer_from_start or getting_closer_backward_from_end else path[:idx+1]
    else:
        return path[:idx+1] if getting_closer_from_start or getting_closer_backward_from_end else path[idx:]

