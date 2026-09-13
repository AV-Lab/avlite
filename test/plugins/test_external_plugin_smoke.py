"""Smoke-import community and member plugins from the two repo-local checkouts."""

from __future__ import annotations

import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PerceptionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import (
    LocalBehavioralPlanningStrategy,
    LocalPathPlanningStrategy,
    LocalPlanningStrategy,
    LocalVelocityPlanningStrategy,
)
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c43_task_strategy import TaskStrategy
from avlite.c60_apps.c61_app_strategy import AppStrategy
from avlite.c60_apps.c63_plugins import (
    import_plugin_modules,
    plugin_module_prefix,
    unregister_plugin_package,
)
from avlite.c60_apps.c68_paths import PluginPaths

_OPTIONAL_EXTRAS = ("carla", "rclpy", "sensor_msgs", "rmw", "librmw")
_SKIP_EXIT = 2


class _Skip(Exception):
    """Worker-only: optional extras missing."""


def _strategy_registries() -> list[dict]:
    return [
        WorldBridge.registry,
        ExecutionStrategy.registry,
        ControlStrategy.registry,
        PerceptionStrategy.registry,
        LocalizationStrategy.registry,
        PredictionStrategy.registry,
        DetectionStrategy.registry,
        TrackingStrategy.registry,
        MappingStrategy.registry,
        GlobalPlannerStrategy.registry,
        LocalPlanningStrategy.registry,
        LocalBehavioralPlanningStrategy.registry,
        LocalPathPlanningStrategy.registry,
        LocalVelocityPlanningStrategy.registry,
        TaskStrategy.registry,
        AppStrategy.registry,
    ]


def _discover_external_plugins() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for root in (PluginPaths.community_dev_dir(), PluginPaths.private_dev_dir()):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
                continue
            if not any(child.rglob("*.py")):
                continue
            found.append((child.name, child))
    return found


def _plugin_cases():
    plugins = _discover_external_plugins()
    if not plugins:
        return [pytest.param(None, None, id="no-external-plugins")]
    return [
        pytest.param(name, path, id=f"{path.parent.name}/{name}")
        for name, path in plugins
    ]


def _registered_classes(plugin_name: str) -> list[type]:
    prefix = plugin_module_prefix(plugin_name)
    seen: set[type] = set()
    classes: list[type] = []
    for registry in _strategy_registries():
        for cls in registry.values():
            if cls in seen or not cls.__module__.startswith(prefix):
                continue
            seen.add(cls)
            classes.append(cls)
    return classes


def _forget_plugin(plugin_name: str) -> None:
    prefix = plugin_module_prefix(plugin_name)
    for registry in _strategy_registries():
        for key, cls in list(registry.items()):
            if getattr(cls, "__module__", "").startswith(prefix):
                del registry[key]
    unregister_plugin_package(plugin_name)


def _is_optional_extra_error(message: str) -> bool:
    lower = message.lower()
    return any(extra.lower() in lower for extra in _OPTIONAL_EXTRAS)


def _load_errors(records) -> list[str]:
    return [
        rec.getMessage()
        for rec in records
        if rec.levelno >= logging.ERROR and "Failed to load" in rec.getMessage()
    ]


def _smoke_one_plugin(plugin_name: str, plugin_dir: Path) -> None:
    _forget_plugin(plugin_name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    handler.setLevel(logging.ERROR)
    plugin_log = logging.getLogger("avlite.c60_apps.c63_plugins")
    plugin_log.addHandler(handler)
    try:
        import_plugin_modules(str(plugin_dir), pkg_name=plugin_name)

        load_errors = _load_errors(records)
        unexpected = [msg for msg in load_errors if not _is_optional_extra_error(msg)]
        if unexpected:
            raise AssertionError(f"{plugin_name} failed to import:\n" + "\n".join(unexpected))

        registered = _registered_classes(plugin_name)
        if not registered:
            if load_errors:
                raise _Skip(f"{plugin_name}: optional extras missing; no strategies registered")
            raise AssertionError(f"{plugin_name}: imported but registered no strategies")

        abstract = [cls.__name__ for cls in registered if inspect.isabstract(cls)]
        if abstract:
            raise AssertionError(f"{plugin_name}: abstract after import: {abstract}")

        settings_mod = sys.modules.get(f"{plugin_module_prefix(plugin_name)}.settings")
        if (plugin_dir / "settings.py").is_file():
            if settings_mod is None:
                raise AssertionError(f"{plugin_name}: settings.py did not import")
            if hasattr(settings_mod, "PluginSettings") and settings_mod.PluginSettings is None:
                raise AssertionError(f"{plugin_name}: PluginSettings is None")
    finally:
        plugin_log.removeHandler(handler)
        _forget_plugin(plugin_name)


@pytest.mark.parametrize("plugin_name,plugin_dir", _plugin_cases())
def test_external_plugin_smoke(plugin_name, plugin_dir):
    if plugin_name is None:
        pytest.skip("no community/private plugin checkouts")

    # Isolate imports: a missing ROS RMW aborts the interpreter (cannot catch).
    env = os.environ.copy()
    root = str(PluginPaths.repo_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", plugin_name, str(plugin_dir)],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return
    if result.returncode == _SKIP_EXIT or _is_optional_extra_error(output):
        pytest.skip(output or f"{plugin_name}: optional extras missing")
    pytest.fail(output or f"{plugin_name}: worker exited {result.returncode}")


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == "--worker":
    try:
        _smoke_one_plugin(sys.argv[2], Path(sys.argv[3]))
    except _Skip as exc:
        print(exc)
        raise SystemExit(_SKIP_EXIT) from exc
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(0)
