from typing import ClassVar

from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


class ExecutionSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c40_execution.yaml"

    c40_executer_type: str = Field(default="SyncExecuter", description="Executer class name.")
    c40_bridge: str = Field(default="BasicSim", description="World bridge class name (e.g. BasicSim, CarlaBridge).")
    c40_perception: str = Field(default="", description="Perception strategy class; empty omits the module.")
    c40_localization: str = Field(default="", description="Localization strategy class; empty omits the module.")
    c40_mapping: str = Field(default="MapReader", description="Mapping strategy class; empty omits the module.")
    c40_global_planner: str = Field(default="GlobalCenterlineRacePlanner", description="Global planner class name; empty omits the module.")
    c40_local_planner: str = Field(default="GreedyLatticePlanner", description="Local planner class name; empty omits the module.")
    c40_controller: str = Field(default="StanleyController", description="Controller class name; empty omits the module.")
    c40_execution_tasks: list[str] = Field(
        default_factory=list,
        description="TaskStrategy class names to append after each stack tick; empty disables tasks.",
    )
    c40_perception_dt: float = Field(default=0.01, description="Perception tick period (seconds).", ge=0.001)
    c40_localization_dt: float = Field(default=0.01, description="Localization tick period (seconds).", ge=0.001)
    c40_replan_dt: float = Field(default=0.01, description="Replanning period (seconds).", ge=0.001)
    c40_control_dt: float = Field(default=0.01, description="Control loop period (seconds).", ge=0.001)
    c40_sim_dt: float = Field(default=0.01, description="Simulation step period (seconds).", ge=0.001)
    c40_global_trajectory: str = Field(default="data/yas_marina_real_race_line_mue_0_5_3_m_margin.json", description="Default global plan JSON path.")
    c40_map: str = Field(
        default="data/race_boundary_yas_marina.map.json",
        description="Map file path (OpenDRIVE .xodr or race-boundary JSON); empty omits the map.",
    )
    c40_reference_point: list[float] | None = Field(
        default_factory=lambda: [24.46992202098782, 54.60522506805341],
        description="WGS84 map origin (lat, lon) in degrees; derived from selected map or set manually.",
    )
    c40_async_combined_perception_planning: bool = Field(default=True, description="Run perception and planning concurrently.")
    c40_log_level: str = Field(default="INFO", description="Python logging level.")
    c40_log_to_file: bool = Field(default=False, description="Write logs to file.")

    c41_world_capabilities: list[str] | None = Field(
        default=None,
        description=(
            "WorldCapability names the bridge feeds into SensorFrame; "
            "null = all advertised world capabilities enabled."
        ),
    )
    c41_world_stack_capabilities: list[str] | None = Field(
        default=None,
        description=(
            "StackCapability names the bridge feeds as ground truth; "
            "null = all advertised world stack capabilities enabled."
        ),
    )

    c46_npc_speed_factor: float = Field(default=0.8, description="NPC speed as fraction of plan speed.")
    c46_npc_control: bool = Field(default=True, description="Enable NPC vehicle controllers in BasicSim.")
    c46_lidar_range: float = Field(default=50.0, description="Simulated LiDAR max range (m).")
    c46_lidar_num_beams: int = Field(default=360, description="Number of simulated LiDAR beams.")
    c46_lidar_fov_deg: float = Field(default=360.0, description="Simulated LiDAR field of view (degrees).")


# Singleton instance: the runtime settings object. Mutated in place by the YAML
# loader and reset helpers — never rebind this name (see settings invariant).
ExecutionSettings = ExecutionSettingsSchema()
