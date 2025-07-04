from c20_planning.c24_global_planners import RaceGlobalPlanner
from c20_planning.c26_local_planners import RNDPlanner
from c30_control.c34_stanley import StanleyController

class ExecutionSettings:
    exclude = []
    filepath: str="configs/c40_execution.yaml"

    async_mode:bool = False
    bridge="Basic" 
    global_planner = RaceGlobalPlanner.__name__
    local_planner = RNDPlanner.__name__
    controller = StanleyController.__name__
    replan_dt=0.5 
    control_dt=0.05
    sim_dt=0.01

    global_trajectory = "data/yas_marina_real_race_line_mue_0_5_3_m_margin.json"
    hd_map = "data/Town10HD_Opt.xodr"

