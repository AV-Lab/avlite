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
    c40_perception_dt: float = Field(
        default=0.01,
        description=(
            "Target perception period (s). Used when c40_pace_perception is true; "
            "ignored for rate when pacing is off (best-effort)."
        ),
        ge=0.001,
    )
    c40_localization_dt: float = Field(default=0.01, description="Localization tick period (seconds).", ge=0.001)
    c40_replan_dt: float = Field(
        default=0.01,
        description=(
            "Target replan period (s). Used when c40_pace_replan is true; "
            "ignored for rate when pacing is off (best-effort)."
        ),
        ge=0.001,
    )
    c40_control_dt: float = Field(
        default=0.01,
        description=(
            "Target control recompute period (s). Used when c40_pace_control is true; "
            "when off, control runs best-effort and the last command is held between recomputes."
        ),
        ge=0.001,
    )
    c40_sim_dt: float = Field(
        default=0.01,
        description=(
            "Fixed simulation integration step (s) when c40_pace_sim is true. "
            "When pacing is off, integration uses clamped wall-clock Δt; this value still caps the max step."
        ),
        ge=0.001,
    )
    c40_pace_perception: bool = Field(
        default=True,
        description="If true, throttle perception to c40_perception_dt; if false, run best-effort every opportunity.",
    )
    c40_pace_replan: bool = Field(
        default=True,
        description="If true, throttle replanning to c40_replan_dt; if false, run best-effort every opportunity.",
    )
    c40_pace_control: bool = Field(
        default=True,
        description=(
            "If true, recompute control at c40_control_dt; if false, recompute best-effort "
            "and hold the last command on simulate steps in between."
        ),
    )
    c40_pace_sim: bool = Field(
        default=True,
        description=(
            "If true, integrate the world with fixed c40_sim_dt and (Sync UI) pad the loop to that period; "
            "if false, do not pad and integrate with clamped wall-clock Δt."
        ),
    )
    c40_global_trajectory: str = Field(default="data/yas_marina_real_race_line_mue_0_5_3_m_margin.json", description="Default global plan JSON path.")
    c40_map: str = Field(
        default="data/race_boundary_yas_marina.map.json",
        description="Map file path (OpenDRIVE .xodr or race-boundary JSON); empty omits the map.",
    )
    c40_reference_point: list[float] | None = Field(
        default_factory=lambda: [24.46992202098782, 54.60522506805341],
        description="WGS84 map origin (lat, lon) in degrees; derived from selected map or set manually.",
    )
    c40_start_pose: list[float] | None = Field(
        default=None,
        description="Ego start pose [x, y, theta]; null starts at the global plan start point.",
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
