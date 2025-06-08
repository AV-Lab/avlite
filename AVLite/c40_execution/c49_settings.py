from c20_planning.c24_global_planners import RaceGlobalPlanner
from c20_planning.c26_local_planners import RNDPlanner
from c30_control.c34_stanley import StanleyController

class ExecutionSettings:
    exclude = []
    filepath: str="configs/c40_execution.yaml"

    # default settings
    profile_name = "default"
    async_mode=False
    bridge="Basic" 
    global_planner = RaceGlobalPlanner.__name__
    local_planner = RNDPlanner.__name__
    controller = StanleyController.__name__
    replan_dt=0.5 
    control_dt=0.05

