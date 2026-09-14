"""Plugin environment: ROS 2 distro and optional WorldBridge launch.sh (c60_apps)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import yaml


class PluginEnv:
    """Source ROS, check a registry entry, and run a world-bridge launch.sh."""

    ORDER = ("foxy", "galactic", "humble", "iron", "jazzy", "kilted", "rolling")
    PREFIX = Path("/opt/ros")
    APPLY_TIMEOUT_S = 15
    launch_sh: ClassVar[Path | None] = None
    _proc: ClassVar[subprocess.Popen | None] = None

    class Error(ValueError):
        """User-facing: why ROS is missing or out of range."""

    def __init__(self) -> None:
        from avlite.c60_apps.c69_settings import AppSettings

        installed: list[str] = []
        if self.PREFIX.is_dir():
            for child in sorted(self.PREFIX.iterdir()):
                if child.is_dir() and (child / "setup.bash").is_file():
                    installed.append(child.name)
        env_distro = os.environ.get("ROS_DISTRO", "").strip()
        if env_distro and env_distro not in installed:
            installed.append(env_distro)
        self.installed = tuple(installed)

        override = (
            os.environ.get("AVLITE_ROS_DISTRO", "").strip()
            or str(getattr(AppSettings, "c60_ros_distro", "") or "").strip()
        )
        if override.lower() in ("", "latest"):
            override = ""
        if override:
            self.active = override
        elif env_distro:
            self.active = env_distro
        else:
            self.active = next(
                (name for name in reversed(self.ORDER) if name in self.installed),
                self.installed[-1] if self.installed else "",
            )

    @property
    def pending_launch(self) -> Path | None:
        script = type(self).launch_sh
        if script is None or not script.is_file():
            return None
        proc = type(self)._proc
        if proc is not None and proc.poll() is None:
            return None
        return script

    def apply(self) -> None:
        """Source setup.bash, sync ``sys.path``, re-exec once for ``LD_LIBRARY_PATH``."""
        if not self.active:
            return
        started_ld = os.environ.get("LD_LIBRARY_PATH", "")
        setup = self.PREFIX / self.active / "setup.bash"
        if setup.is_file():
            try:
                proc = subprocess.run(
                    ["bash", "-c", f'source "{setup}" && env -0'],
                    check=True,
                    capture_output=True,
                    timeout=self.APPLY_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                raise self.Error(
                    f"Timed out sourcing ROS 2 {self.active} after {self.APPLY_TIMEOUT_S}s"
                ) from exc
            except (OSError, subprocess.CalledProcessError) as exc:
                raise self.Error(f"Could not source ROS 2 {self.active}: {exc}") from exc
            for item in proc.stdout.split(b"\0"):
                if not item or b"=" not in item:
                    continue
                key, _, val = item.partition(b"=")
                name = key.decode("utf-8", "surrogateescape")
                if not name or name.startswith("AVLITE_"):
                    continue
                os.environ[name] = val.decode("utf-8", "surrogateescape")
        self._sync_sys_path()
        if not setup.is_file() or os.environ.get("AVLITE_ROS_SOURCED") == self.active:
            return
        needed = [
            path
            for path in (
                self.PREFIX / self.active / "lib",
                self.PREFIX / self.active / "lib" / "x86_64-linux-gnu",
            )
            if path.is_dir()
        ]
        parts = started_ld.split(os.pathsep)
        if not needed or any(str(path) in parts for path in needed):
            return
        os.environ["AVLITE_ROS_SOURCED"] = self.active
        if self._running_under_pytest():
            return
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

    @staticmethod
    def _running_under_pytest() -> bool:
        return bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules

    @staticmethod
    def _sync_sys_path() -> None:
        """Prepend ``PYTHONPATH`` entries so this process can import rclpy."""
        new = [
            part
            for part in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if part and part not in sys.path
        ]
        for part in reversed(new):
            sys.path.insert(0, part)

    def check(self, entry: dict) -> None:
        """Raise Error if entry.require_ros and active is missing/out of range."""
        if not entry.get("require_ros"):
            return
        label = str(entry.get("display_name") or entry.get("name") or "This plugin").strip()
        min_ver = str(entry.get("min_ros_version") or "").strip().lower()
        max_ver = str(entry.get("max_ros_version") or "").strip().lower()
        if min_ver and max_ver:
            needed = f"{min_ver}–{max_ver}"
        elif min_ver:
            needed = f"{min_ver}+"
        elif max_ver:
            needed = f"≤{max_ver}"
        else:
            needed = "any"
        found = ", ".join(self.installed) if self.installed else "(none)"
        active = self.active or "none"
        hint = (
            f"{label} needs ROS 2 {needed}. Active distro: {active}. "
            f"Installed: {found}. Install ROS 2 or set c60_ros_distro."
        )
        if not self.active:
            raise self.Error(hint)
        rank = {name: i for i, name in enumerate(self.ORDER)}
        current = rank.get(self.active.lower())
        if current is None:
            raise self.Error(hint)
        if min_ver and current < rank.get(min_ver, current):
            raise self.Error(hint)
        if max_ver and current > rank.get(max_ver, current):
            raise self.Error(hint)

    def bind(self, world: Any) -> None:
        """Check the selected world plugin and remember launch.sh if present."""
        from avlite.c60_apps.c63_plugins import plugin_module_prefix
        from avlite.c60_apps.c68_paths import PluginPaths
        from avlite.c60_apps.c69_settings import AppSettings

        type(self).launch_sh = None
        module = type(world).__module__
        for name, stored in AppSettings.c62_community_plugins.items():
            prefix = plugin_module_prefix(name)
            if not module.startswith(prefix):
                continue
            path = PluginPaths.load_path(name, stored)
            if path is None:
                return
            meta_path = path / ".avlite-registry.yaml"
            if meta_path.is_file():
                entry = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                if isinstance(entry, dict):
                    self.check(entry)
            script = path / "launch.sh"
            if script.is_file():
                type(self).launch_sh = script
            return

    def start_launch(self) -> None:
        """Run launch.sh in the background if it is not already running."""
        script = self.pending_launch
        if script is None:
            return
        type(self)._proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(script.parent),
            start_new_session=True,
        )

    def launch_warning(self) -> str:
        script = self.pending_launch or type(self).launch_sh
        return (
            f"This plugin will run launch.sh from {script}. "
            "It may start a simulator or vehicle platform and will keep running after Stop."
        )
