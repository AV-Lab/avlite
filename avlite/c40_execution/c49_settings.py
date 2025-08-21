
# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
from avlite.c20_planning.c24_global_planners import RaceGlobalPlanner
from avlite.c20_planning.c26_local_planners import GreedyLatticePlanner
from avlite.c30_control.c34_stanley import StanleyController

class ExecutionSettings:
    exclude = ["exclude"]
    filepath: str="configs/c40_execution.yaml"

    # executer_type = SyncExecuter.__name__  
    async_mode:bool = False
    bridge="BasicSim" # Options: Basic, Carla, Gazebo, ROS
    perception = ""
    global_planner = RaceGlobalPlanner.__name__
    local_planner = GreedyLatticePlanner.__name__
    controller = StanleyController.__name__
    replan_dt=0.5 
    control_dt=0.05
    sim_dt=0.01

    global_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    hd_map = "data/san_campus.xodr"

    community_extensions: dict[str,str] = {"delete_me": "/home/mkhonji/Dropbox/20-development/21-software-dev/21.2-AVlite/avlite-plugins/delete_me"}
    default_extensions: list[str] = []

    basic_sim_default_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    basic_sim_npc_speed_factor = 0.8   
    basic_sim_npc_control = True  # If True, NPCs will follow the default trajectory at the above speed factor


