from os import wait
import networkx as nx
import numpy as np
import logging
from scipy.signal import savgol_filter

from c10_perception.c18_hdmap import HDMap
from c20_planning.c21_planning_model import GlobalPlan
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c28_trajectory import Trajectory, convert_sd_path_to_xy_path

log = logging.getLogger(__name__)

class RaceGlobalPlanner(GlobalPlannerStrategy):
    def __init__(self):
        super().__init__()


    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass

# Assumtpions
# TODO: Lane sections are not handled properly
# TODO: Assumptions: lanes of the same road are assumed to have the same size
class HDMapGlobalPlanner(GlobalPlannerStrategy):
    """
    A global planner that uses OpenDRIVE HD maps for path planning.
    """

    def __init__(self, xodr_file:str, sampling_resolution=.5, max_velocity=10, wp_to_full_velocity=20):
        """
        :param xodr_file: path to the OpenDRIVE HD map (.xodr).
        :param sampling_resolution: distance (meters) between samples when converting arcs/lines to discrete points.
        :param max_velocity: maximum velocity for the path.
        """
        super().__init__()
        self.xodr_file = xodr_file
        self.sampling_resolution = sampling_resolution

        self.hdmap:HDMap = HDMap(xodr_file_name=xodr_file, sampling_resolution=sampling_resolution)
        self.max_velocity = max_velocity
        self.wp_to_full_velocity = wp_to_full_velocity
        
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
        if len(path) == 0:
            log.warning(f"No path found from {start_lane.uid} to {goal_lane.uid}.")
            return

        # removing lane changes by keeping only the final lane in that road. This will utilize lane_change attribute in the edge
        # Notice that nodes and lanes, and edges are the changes
        # TODO: Lane sections are not handled properly
        # TODO: Assumptions: lanes of the same road are assumed to have the same size
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
        if len(path2) == 1:
            lane = self.hdmap.lane_by_uid[path2[0]]
            chopped_path = chop_path_from_two_sides(lane.center_line.T, lane.id, s_idx, g_idx)
            for point in chopped_path:
                self.global_plan.path.append(point)
                self.global_plan.left_boundary_d.append(lane.width/2)
                self.global_plan.right_boundary_d.append(-lane.width/2)
                self.global_plan.velocity.append(self.max_velocity)
        else:
            for i, uid in enumerate(path2):
                lane = self.hdmap.lane_by_uid[uid]
                neighbor_lanes = [self.hdmap.lane_by_uid[uid] for uid in self.hdmap.lane_network.neighbors(lane.uid)]
                same_road_lanes = [l for l in neighbor_lanes if l.road_id == lane.road_id]
                if int(lane.id) < 0:
                    lane_count_to_right = sum(1 for l in same_road_lanes if int(l.id) < int(lane.id))
                    lane_count_to_left = sum(1 for l in same_road_lanes if int(l.id) > int(lane.id))
                else:
                    lane_count_to_right = sum(1 for l in same_road_lanes if int(l.id) > int(lane.id))
                    lane_count_to_left = sum(1 for l in same_road_lanes if int(l.id) < int(lane.id))
                
                log.debug(f"Lane {lane.uid} has neighbors: {[l.uid for l in same_road_lanes]}, with {lane_count_to_left} lanes to the left and {lane_count_to_right} lanes to the right.")


                if i == 0:
                    log.debug(f"Start lane: {lane.uid}, point IDX: {s_idx}")
                    for point in chop_path(lane.center_line.T, lane.id, s_idx, start=True):
                        self.global_plan.path.append(point)
                        self.global_plan.left_boundary_d.append(lane.width/2 * lane_count_to_left +  lane.width/2)
                        self.global_plan.right_boundary_d.append(-lane.width/2 - lane.width/2 * lane_count_to_right)
                        
                        self.global_plan.lane_left_boundary_d.append(lane.width/2)
                        self.global_plan.lane_right_boundary_d.append(-lane.width/2)
                elif i == len(path2) - 1:
                    log.debug(f"Goal lane: {lane.uid}, point IDX: {g_idx}")
                    for point in chop_path(lane.center_line.T, lane.id, g_idx, start=False):
                        self.global_plan.path.append(point)
                        self.global_plan.left_boundary_d.append(lane.width/2 * lane_count_to_left +  lane.width/2)
                        self.global_plan.right_boundary_d.append(-lane.width/2 - lane.width/2 * lane_count_to_right)
                        
                        self.global_plan.lane_left_boundary_d.append(lane.width/2)
                        self.global_plan.lane_right_boundary_d.append(-lane.width/2)
                else:
                    lane_path = lane.center_line.T if int(lane.id) < 0 else lane.center_line.T[::-1]
                    for point in lane_path:
                        self.global_plan.path.append(point)
                        self.global_plan.left_boundary_d.append(lane.width/2 * lane_count_to_left +  lane.width/2)
                        self.global_plan.right_boundary_d.append(-lane.width/2 - lane.width/2 * lane_count_to_right)

                        self.global_plan.lane_left_boundary_d.append(lane.width/2)
                        self.global_plan.lane_right_boundary_d.append(-lane.width/2)
       
        # setting velocity. It should start with zero to max vel for the first 20 points, then at max, then reduce to zero last 20 points
        n = len(self.global_plan.path)
        if n < self.wp_to_full_velocity * 2:
            self.global_plan.velocity = [self.max_velocity] * n
        else:
            self.global_plan.velocity = list(np.linspace(0, self.max_velocity, self.wp_to_full_velocity)) + [self.max_velocity] * (n - 40) + \
                list(np.linspace(self.max_velocity, 0, self.wp_to_full_velocity))

        # self.global_plan = remove_dublicate_points(self.global_plan)
        # self.global_plan = smoothen_path_savgol(self.global_plan, min_spacing=0.5, window_length=7, polyorder=3)
        self.global_plan = smoothen_path_splprep(self.global_plan, min_spacing=0.5, smoothing=20)
        
        self.global_plan.lane_path = [self.hdmap.lane_by_uid[lane_uid] for lane_uid in path2]
        self.global_plan.trajectory = Trajectory(path=self.global_plan.path, velocity=self.global_plan.velocity)
        self.global_plan.left_boundary_x,self.global_plan.left_boundary_y = convert_sd_path_to_xy_path(
                self.global_plan.trajectory,
                self.global_plan.trajectory.path_s,
                self.global_plan.left_boundary_d)
        self.global_plan.right_boundary_x, self.global_plan.right_boundary_y = convert_sd_path_to_xy_path(
                self.global_plan.trajectory, self.global_plan.trajectory.path_s, self.global_plan.right_boundary_d)

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



def chop_path(path, lane_id, idx, start=True):
    """
    Chops the path at a given index, either from the start or end.
    """
    is_neg = int(lane_id) < 0
    log.debug(f"Chopping path at index {idx} for lane {lane_id} (start={start}, is_neg={is_neg})")
    if start and is_neg:
        path = path[idx:]
    elif start and not is_neg:
        path = path[:idx+1]
    elif not start and is_neg:
        path = path[:idx+1] 
    elif not start and not is_neg:
        path = path[idx:]

    path = path if int(lane_id) < 0 else path[::-1]
    return path




def chop_path_from_two_sides(path, lane_id, s_idx,g_idx):
    is_neg = int(lane_id) < 0
    log.debug(f"Chopping path from two sides for lane {lane_id} (s_idx={s_idx}, g_idx={g_idx}, is_neg={is_neg})")
    if is_neg:
        path = path[s_idx:g_idx+1]
    else:
        path = path[g_idx:s_idx+1]

    path = path if int(lane_id) < 0 else path[::-1]
    return path

def remove_dublicate_points(plan: GlobalPlan):
    """
    removes duplicate points from the global plan path.
    """
    # Build lists of indices to keep (avoiding deletion issues)
    indices_to_keep = []
    for i in range(1,len(plan.path)):
        # if points are reasonably different, but not too fa
        if not np.array_equal(plan.path[i], plan.path[i-1]):
            indices_to_keep.append(i)
        else:
            log.warning(f"Removed duplicate point {plan.path[i]} from global plan path.")
    
    plan.path = [plan.path[i] for i in indices_to_keep]
    plan.velocity = [plan.velocity[i] for i in indices_to_keep]
    plan.left_boundary_d = [plan.left_boundary_d[i] for i in indices_to_keep]
    plan.right_boundary_d = [plan.right_boundary_d[i] for i in indices_to_keep]
    
    return plan
from scipy.signal import savgol_filter
import numpy as np

def smoothen_path_savgol(plan: GlobalPlan, min_spacing=0.5, window_length=7, polyorder=3):
    """
    Removes near-duplicate points from the global plan path and applies smoothing.
    """
    if len(plan.path) < 2:
        return plan

    cleaned_path = [plan.path[0]]
    cleaned_velocity = [plan.velocity[0]]
    cleaned_left_d = [plan.left_boundary_d[0]]
    cleaned_right_d = [plan.right_boundary_d[0]]
    for i in range(1, len(plan.path)):
        if np.linalg.norm(np.array(plan.path[i]) - np.array(cleaned_path[-1])) > min_spacing:
            cleaned_path.append(plan.path[i])
            cleaned_velocity.append(plan.velocity[i])
            cleaned_left_d.append(plan.left_boundary_d[i])
            cleaned_right_d.append(plan.right_boundary_d[i])
        else:
            log.warning(f"Removed near-duplicate point {plan.path[i]} at index {i} from global plan path.")

    n = len(cleaned_path)
    if n >= 3:
        if window_length >= n:
            window_length = n // 2 * 2 + 1  # Make it a valid odd number
        path_np = np.array(cleaned_path)
        x_smooth = savgol_filter(path_np[:, 0], window_length, polyorder, mode="interp")
        y_smooth = savgol_filter(path_np[:, 1], window_length, polyorder, mode="interp")
        cleaned_path = list(zip(x_smooth, y_smooth))

    plan.path = cleaned_path
    plan.velocity = cleaned_velocity
    plan.left_boundary_d = cleaned_left_d
    plan.right_boundary_d = cleaned_right_d

    return plan

from scipy.interpolate import splprep, splev
import numpy as np

def smoothen_path_splprep(plan: GlobalPlan, min_spacing=0.5, smoothing=0.5):
    """
    Removes near-duplicate points from the global plan path and applies B-spline smoothing.
    """
    if len(plan.path) < 2:
        return plan

    cleaned_path = [plan.path[0]]
    cleaned_velocity = [plan.velocity[0]]
    cleaned_left_d = [plan.left_boundary_d[0]]
    cleaned_right_d = [plan.right_boundary_d[0]]
    for i in range(1, len(plan.path)):
        if np.linalg.norm(np.array(plan.path[i]) - np.array(cleaned_path[-1])) >= min_spacing:
            cleaned_path.append(plan.path[i])
            cleaned_velocity.append(plan.velocity[i])
            cleaned_left_d.append(plan.left_boundary_d[i])
            cleaned_right_d.append(plan.right_boundary_d[i])
        else:
            log.debug(f"Removed near-duplicate point {plan.path[i]} from global plan path.")

    # Apply B-spline smoothing to path coordinates
    cleaned_path_np = np.array(cleaned_path)
    if len(cleaned_path_np) >= 4:  # B-spline requires at least 4 points for cubic smoothing
        # Fit spline to x and y separately, using arc-length as parameter
        tck, _ = splprep([cleaned_path_np[:, 0], cleaned_path_np[:, 1]], s=smoothing)
        u_new = np.linspace(0, 1, len(cleaned_path_np))
        x_smooth, y_smooth = splev(u_new, tck)
        cleaned_path = list(zip(x_smooth, y_smooth))

    # Step 3: Update plan fields
    plan.path = cleaned_path
    plan.velocity = cleaned_velocity
    plan.left_boundary_d = cleaned_left_d
    plan.right_boundary_d = cleaned_right_d

    return plan

