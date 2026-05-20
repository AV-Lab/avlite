class ExtensionSettings:
    exclude = ["exclude", "filepath"]  # attributes to exclude from saving/loading
    filepath: str = "configs/ext_headless_mode.yaml"

    # Maximum number of log lines kept in the dashboard buffer
    log_buffer_size: int = 500

    # Terminal dashboard refresh rate (Hz)
    dashboard_refresh_hz: float = 10.0

    # Number of rows reserved for the stats panel in the dashboard layout
    stats_panel_height: int = 18
