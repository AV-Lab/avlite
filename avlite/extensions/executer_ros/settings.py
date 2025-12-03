class ExtensionSettings:
    exclude = ["exclude", "filepath"] # attributes to exclude from saving/loading
    filepath: str ="configs/ext_ros_executer.yaml"

    perception_topic: str = "perception topic"
    global_Planner_topic: str = "global planner topic"
    local_planner_topic: str = "local planner"
    controller_topic: str = "controller topic"


