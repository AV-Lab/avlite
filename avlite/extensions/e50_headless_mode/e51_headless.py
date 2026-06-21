"""AVLite headless-mode extension.

Provides a terminal dashboard (via ``rich``) that runs the executer in a
background thread and displays live statistics and logs.
"""

import argparse
import logging
import sys

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _DequeLogHandler(logging.Handler):
    """Logging handler that appends formatted records into a bounded deque."""

    def __init__(self, buffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record):
        try:
            self._buffer.append(self.format(record))
        except Exception:
            pass


class _FDStreamCapture:
    """Redirect a real OS file descriptor (e.g. fd 2 / stderr) into a deque.

    This catches output from C/C++ libraries (rclpy/rcutils, native code) that
    write directly to fd 2 and would otherwise corrupt rich.live's display.
    """

    def __init__(self, fd: int, buffer, prefix: str = ""):
        self._fd = fd
        self._buffer = buffer
        self._prefix = prefix
        self._saved_fd = None
        self._read_fd = None
        self._write_fd = None
        self._thread = None
        self._stop = False

    def start(self) -> None:
        import os
        import threading

        self._saved_fd = os.dup(self._fd)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, self._fd)
        os.close(self._write_fd)
        self._write_fd = None

        def _pump():
            buf = b""
            while not self._stop:
                try:
                    chunk = os.read(self._read_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").rstrip("\r")
                    if text:
                        self._buffer.append(self._prefix + text)
            if buf:
                text = buf.decode("utf-8", errors="replace")
                if text:
                    self._buffer.append(self._prefix + text)

        self._thread = threading.Thread(target=_pump, name=f"fd{self._fd}-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        import os

        self._stop = True
        if self._saved_fd is not None:
            try:
                os.dup2(self._saved_fd, self._fd)
                os.close(self._saved_fd)
            except OSError:
                pass
            self._saved_fd = None
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
            self._read_fd = None


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def _render_dashboard(executer, profile: str):
    """Build a ``rich`` renderable summarising the live executer state (stats only)."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _g(obj, attr, default="-"):
        try:
            val = getattr(obj, attr)
            return val if val is not None else default
        except Exception:
            return default

    def _fmt(v, spec=""):
        try:
            return format(v, spec)
        except Exception:
            return str(v)

    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white", no_wrap=True, justify="left")

    ego = _g(executer, "ego_state", None)
    controller = _g(executer, "controller", None)
    cmd = _g(controller, "cmd", None) if controller != "-" else None
    local_planner = _g(executer, "local_planner", None)
    world = _g(executer, "world", None)

    table.add_row("Profile", str(profile))
    requested = getattr(executer, "_requested_executer_type", None)
    actual = type(executer).__name__
    if requested and requested != actual:
        table.add_row("Executer", f"{actual}  [yellow](requested: {requested})[/yellow]")
    else:
        table.add_row("Executer", actual)
    table.add_row("World", type(world).__name__ if world not in (None, "-") else "-")
    table.add_row(
        "Elapsed (real / sim)",
        f"{_fmt(_g(executer, 'elapsed_real_time', 0.0), '<7.2f')} s  /  "
        f"{_fmt(_g(executer, 'elapsed_sim_time', 0.0), '<7.2f')} s",
    )

    def _fps_cell(fps, target_dt):
        """Format FPS with its target rate. Colour green if near target, yellow if slow."""
        fps_val = fps if isinstance(fps, (int, float)) else 0.0
        target = target_dt if isinstance(target_dt, (int, float)) and target_dt > 0 else None
        target_fps = 1.0 / target if target else None
        fps_str = f"{fps_val:<6.1f}"
        if target_fps:
            target_str = f"[dim]/{target_fps:<5.1f}[/dim]"
        else:
            target_str = ""
        if target_fps and fps_val >= target_fps * 0.9:
            return f"[green]{fps_str}[/green]{target_str}"
        elif fps_val > 0:
            return f"[yellow]{fps_str}[/yellow]{target_str}"
        else:
            return f"[red]{fps_str}[/red]{target_str}"

    table.add_row(
        "FPS  (actual[dim]/target[/dim])",
        f"plan  {_fps_cell(_g(executer, 'planner_fps', 0.0), _g(executer, 'replan_dt', None))}  "
        f"ctrl  {_fps_cell(_g(executer, 'control_fps', 0.0), _g(executer, 'control_dt', None))}  "
        f"perc  {_fps_cell(_g(executer, 'perception_fps', 0.0), _g(executer, 'perception_dt', None))}  "
        f"loc  {_fps_cell(_g(executer, 'localization_fps', 0.0), _g(executer, 'localization_dt', None))}",
    )
    if ego not in (None, "-"):
        table.add_row(
            "Ego (x, y, θ)",
            f"({_fmt(_g(ego, 'x', 0.0), '<+8.2f')}, "
            f"{_fmt(_g(ego, 'y', 0.0), '<+8.2f')}, "
            f"{_fmt(_g(ego, 'theta', 0.0), '<+6.2f')})",
        )
        v = _g(ego, "velocity", 0.0)
        table.add_row(
            "Velocity",
            f"{_fmt(v, '<6.2f')} m/s  ({_fmt(v * 3.6 if isinstance(v, (int, float)) else 0.0, '<6.2f')} km/h)",
        )
    if local_planner not in (None, "-"):
        table.add_row("Lap", str(_g(local_planner, "lap", 0)))
    if cmd not in (None, "-"):
        table.add_row(
            "Last cmd (acc / steer)",
            f"{_fmt(_g(cmd, 'acceleration', 0.0), '<+6.2f')}  /  "
            f"{_fmt(_g(cmd, 'steer', 0.0), '<+6.2f')}",
        )

    footer = Text("Press Ctrl+C to stop", style="dim")
    body = Table.grid(expand=True)
    body.add_row(table)
    body.add_row(footer)
    return Panel(body, title="[bold]AVlite — Headless[/bold]", border_style="green")


def _render_log_panel(log_buffer, height: int):
    from rich.panel import Panel
    from rich.text import Text

    # Show the most recent lines that fit in the panel.
    recent = list(log_buffer)[-height:]
    if not recent:
        body = Text("(no logs yet)", style="dim")
    else:
        body = Text()
        level_styles = {
            "DEBUG": "dim",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        for line in recent:
            style = "white"
            for lvl, s in level_styles.items():
                if f" {lvl} " in line:
                    style = s
                    break
            body.append(line + "\n", style=style)
    return Panel(body, title="[bold]Logs[/bold]", border_style="blue", height=height + 2)


def _build_layout(executer, profile, log_buffer, log_height: int, stats_panel_height: int):
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(_render_log_panel(log_buffer, log_height), name="logs"),
        Layout(_render_dashboard(executer, profile), name="stats", size=stats_panel_height),
    )
    return layout


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _strip_console_handlers(logger: logging.Logger) -> None:
    """Remove StreamHandlers (but not FileHandlers) from *logger*."""
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            logger.removeHandler(h)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def register_parser(subparsers) -> None:
    """Add the ``headless`` sub-command to *subparsers*."""
    p = subparsers.add_parser("headless", help="Run the executer headless with a terminal dashboard")
    p.add_argument("-p", "--profile", default=None, help="Profile name to load (default: 'default')")
    p.add_argument("profile_pos", nargs="?", default=None, help=argparse.SUPPRESS)
    p.add_argument("--control-dt", type=float, default=None, help="Control loop dt in seconds (default: from profile)")
    p.add_argument("--replan-dt", type=float, default=None, help="Replan dt in seconds (default: from profile)")
    p.add_argument("--perceive", action="store_true", help="Enable perception step in the loop")
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level from the profile (default: read from profile)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_headless(profile: str, control_dt: float, replan_dt: float, perceive: bool, log_level: str | None) -> None:
    """Run the executer in a background thread with a live ``rich`` dashboard."""
    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        sys.stderr.write(
            "Headless mode requires the 'rich' package.\n"
            "Install it with:  pip install rich\n"
        )
        sys.exit(1)

    import os as _os
    import threading
    import time
    from collections import deque
    from datetime import datetime
    from pathlib import Path

    from avlite.c40_execution.c43_factory import executor_factory
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_common.c69_setting_utils import load_all_stack_settings
    from avlite.extensions.e50_headless_mode.settings import ExtensionSettings

    # Use INFO temporarily until the profile is loaded and the real level is known.
    level_value = logging.INFO

    # Strip every console StreamHandler attached to root *and* every existing
    # non-root logger; otherwise libraries that grabbed sys.stderr at import
    # time will continue to write past rich.live's redirect.
    root_logger = logging.getLogger()
    _strip_console_handlers(root_logger)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lg = logging.getLogger(name)
        if isinstance(lg, logging.Logger):
            _strip_console_handlers(lg)

    # Placeholder deque; resized to ExtensionSettings.log_buffer_size after settings load.
    log_buffer: deque[str] = deque(maxlen=500)

    deque_handler = _DequeLogHandler(log_buffer)
    deque_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    deque_handler.setLevel(level_value)
    root_logger.addHandler(deque_handler)
    root_logger.setLevel(level_value)

    # Redirect OS-level stderr (fd 2) so C-library output (rclpy/rcutils, etc.)
    # ends up in the log buffer instead of corrupting the live display.
    fd_capture = _FDStreamCapture(2, log_buffer, prefix="[stderr] ")
    fd_capture.start()

    # Nudge ROS to be quieter at the source, if the user hasn't set these.
    _os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "0")
    _os.environ.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
    _os.environ.setdefault("RCUTILS_LOGGING_MIN_SEVERITY", (log_level or "INFO").upper())

    console = Console(stderr=False)

    load_all_stack_settings(profile=profile, load_extensions=True)

    # Re-create the buffer with the configured capacity now that settings are loaded.
    log_buffer = deque(log_buffer, maxlen=ExtensionSettings.log_buffer_size)
    deque_handler._buffer = log_buffer
    fd_capture._buffer = log_buffer

    # CLI args win over profile; profile wins over built-in defaults.
    if control_dt is None:
        control_dt = ExecutionSettings.c40_control_dt
    if replan_dt is None:
        replan_dt = ExecutionSettings.c40_replan_dt

    # Resolve effective log level: CLI arg wins over profile; profile wins over default.
    effective_log_level = log_level if log_level is not None else ExecutionSettings.c40_log_level
    level_value = getattr(logging, effective_log_level.upper(), logging.INFO)
    root_logger.setLevel(level_value)
    deque_handler.setLevel(level_value)

    # Attach a file handler if the profile requests it.
    file_handler: logging.FileHandler | None = None
    if ExecutionSettings.c40_log_to_file:
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"avlite_{timestamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(lineno)-4d [%(levelname).4s] %(name)-36s: %(message)s"
            )
        )
        file_handler.setLevel(level_value)
        root_logger.addHandler(file_handler)
        log.info(f"Logging to file: {log_path}")

    executer = executor_factory(
        executer_type=ExecutionSettings.c40_executer_type,
        bridge=ExecutionSettings.c40_bridge,
        perception_strategy_name=ExecutionSettings.c40_perception,
        localization_strategy_name=ExecutionSettings.c40_localization,
        global_planner_strategy_name=ExecutionSettings.c40_global_planner,
        local_planner_strategy_name=ExecutionSettings.c40_local_planner,
        controller_strategy_name=ExecutionSettings.c40_controller,
        perception_dt=ExecutionSettings.c40_perception_dt,
        localization_dt=ExecutionSettings.c40_localization_dt,
        replan_dt=replan_dt,
        control_dt=control_dt,
        hd_map=ExecutionSettings.c40_hd_map,
        default_global_trajectory_file=ExecutionSettings.c40_global_trajectory,
        load_extensions=True,
        async_combined_perception_planning=ExecutionSettings.c40_async_combined_perception_planning,
    )

    # Re-strip handlers that may have been added during factory/extension import.
    _strip_console_handlers(root_logger)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lg = logging.getLogger(name)
        if isinstance(lg, logging.Logger):
            _strip_console_handlers(lg)
    if deque_handler not in root_logger.handlers:
        root_logger.addHandler(deque_handler)

    runner = threading.Thread(
        target=executer.run,
        kwargs={
            "replan_dt": replan_dt,
            "control_dt": control_dt,
            "call_replan": True,
            "call_control": True,
            "call_perceive": perceive,
        },
        daemon=True,
        name="avlite-headless-runner",
    )
    runner.start()

    stats_panel_height = ExtensionSettings.stats_panel_height

    def _log_height() -> int:
        # Reserve rows for the stats panel + 4 for the log panel border/title.
        return max(5, console.size.height - stats_panel_height - 4)

    try:
        with Live(
            _build_layout(executer, profile, log_buffer, _log_height(), stats_panel_height),
            console=console,
            refresh_per_second=ExtensionSettings.dashboard_refresh_hz,
            screen=True,
            redirect_stdout=True,
            redirect_stderr=True,
        ) as live:
            while runner.is_alive():
                time.sleep(0.1)
                live.update(_build_layout(executer, profile, log_buffer, _log_height(), stats_panel_height))
    except KeyboardInterrupt:
        try:
            executer.stop()
        except Exception:
            pass
        console.print("[yellow]Stopped.[/yellow]")
    finally:
        root_logger.removeHandler(deque_handler)
        if file_handler is not None:
            root_logger.removeHandler(file_handler)
            file_handler.close()
        fd_capture.stop()


