"""Plugin settings for the Qt5 visualizer (no Qt import at module load)."""

from __future__ import annotations

from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    p60_dark_mode: bool = Field(default=True, description="Use dark UI theme.")
    p60_shortcut_mode: bool = Field(default=False, description="Collapse stack panels; show shortcut help.")
    p60_next_profile: str = Field(default="default", description="Profile switched to with shortcut F.")
    p60_bg_color: str = Field(default="#1e1e1e", description="Plot background color.")
    p60_fg_color: str = Field(default="#dddddd", description="Plot foreground/text color.")
    p60_ui_refresh_hz: float = Field(
        default=60.0, ge=5.0, le=240.0,
        description="UI/plot refresh rate (Hz). Sim/planning stay off the UI thread.",
    )
    p60_use_opengl: bool = Field(
        default=True,
        description="Enable pyqtgraph OpenGL acceleration when available.",
    )

    p66_show_legend: bool = Field(default=False, description="Show plot legend.")
    p66_show_past_locations: bool = Field(default=True, description="Show historical ego positions.")
    p66_show_global_plan: bool = Field(default=True, description="Draw global plan.")
    p66_show_global_plan_boundaries: bool = Field(default=True, description="Show left/right plan boundaries.")
    p66_show_local_plan: bool = Field(default=True, description="Draw local plan.")
    p66_show_local_lattice: bool = Field(default=True, description="Draw local lattice.")
    p66_show_state: bool = Field(default=True, description="Show ego/agent boxes.")
    p66_global_view_follow_planner: bool = Field(default=False, description="Follow planner in global XY.")
    p66_frenet_view_follow_planner: bool = Field(default=False, description="Follow planner in Frenet.")
    p66_show_lidar_global: bool = Field(default=True, description="Show LiDAR in XY view.")
    p66_show_lidar_frenet: bool = Field(default=False, description="Show LiDAR in Frenet view.")
    p66_show_lidar_clusters: bool = Field(default=True, description="Highlight LiDAR clusters.")
    p66_show_race_boundary: bool = Field(default=True, description="Show race/track boundary.")
    p66_xy_zoom: float = Field(default=30.0, description="Local XY zoom (half-width, m).")
    p66_frenet_zoom: float = Field(default=30.0, description="Frenet zoom (half-width, m).")
    p66_global_zoom: float = Field(default=80.0, description="Global plot zoom (half-width, m).")

    p67_show_local_global_view: bool = Field(default=True, description="Show local XY sub-view.")
    p67_show_local_frenet_view: bool = Field(default=True, description="Show local Frenet sub-view.")
    p67_show_occupancy_map: bool = Field(default=False, description="Show occupancy map overlay.")
    p67_show_prediction: bool = Field(default=True, description="Show predicted trajectories.")
    p67_global_plan_view: bool = Field(default=True, description="Show global plan panel.")
    p67_local_plan_view: bool = Field(default=True, description="Show local plan panel.")

    p68_show_core_logs: bool = Field(default=True, description="Show core module logs.")
    p68_show_perceive_logs: bool = Field(default=True, description="Show perception logs.")
    p68_show_plan_logs: bool = Field(default=True, description="Show planning logs.")
    p68_show_control_logs: bool = Field(default=True, description="Show control logs.")
    p68_show_execute_logs: bool = Field(default=True, description="Show execution logs.")
    p68_show_plugins_logs: bool = Field(default=True, description="Show plugin logs.")
    p68_disable_log: bool = Field(default=False, description="Disable log panel updates.")
    p68_max_log_lines: int = Field(default=1000, description="Max log lines retained.")
    p68_log_pull_time: int = Field(default=50, description="Log refresh interval (ms).")


PluginSettings = PluginSettingsSchema()
