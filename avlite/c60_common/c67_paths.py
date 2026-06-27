"""Filesystem paths for AVLite configs, data, and community plugins."""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_PLUGINS_SUBDIR = Path("avlite") / "plugins"


def _xdg_config_base() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    return Path(xdg).expanduser() if xdg else Path.home() / ".config"


def get_config_dir() -> Path:
    """User AVLite config root (~/.config/avlite by default)."""
    env = os.environ.get("AVLITE_CONFIG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_xdg_config_base() / "avlite").resolve()


def _user_meta_dir() -> Path:
    """Preferences always under ~/.config/avlite (ignores ``AVLITE_CONFIG_DIR``)."""
    return (_xdg_config_base() / "avlite").resolve()


def bundled_config_dir() -> Path:
    """Shipped YAML directory (git clone ``configs/``); missing after plain pip install."""
    return Path(__file__).resolve().parent.parent.parent / "configs"


def get_repo_config_dir() -> Path:
    """Repository shipped defaults (…/configs)."""
    return bundled_config_dir()


def can_edit_repo_configs() -> bool:
    d = bundled_config_dir()
    return d.is_dir() and any(d.glob("*.yaml"))


def is_repo_config_target() -> bool:
    path = _user_meta_dir() / "config_target"
    if not path.is_file():
        return False
    return path.read_text(encoding="utf-8").strip() == "repo"


def set_repo_config_target(enabled: bool) -> None:
    _user_meta_dir().mkdir(parents=True, exist_ok=True)
    value = "repo" if enabled else "user"
    (_user_meta_dir() / "config_target").write_text(value + "\n", encoding="utf-8")


def get_user_configs_dir() -> Path:
    """User YAML profiles directory (same as ``get_config_dir()``)."""
    return get_config_dir()


def effective_config_path(stored_filepath: str, *, for_write: bool = False) -> str:
    """Resolve a settings filepath: user config dir for writes; user if present else repo for reads."""
    path = Path(stored_filepath)
    if path.is_absolute():
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    name = path.name
    user = get_config_dir() / name
    repo = get_repo_config_dir() / name
    if is_repo_config_target() and can_edit_repo_configs():
        bundled = get_repo_config_dir() / name
        if for_write:
            bundled.parent.mkdir(parents=True, exist_ok=True)
        return str(bundled)
    if for_write:
        get_config_dir().mkdir(parents=True, exist_ok=True)
        return str(user)
    return str(user if user.is_file() else repo)


def clear_user_configs() -> list[str]:
    """Remove user-local stack YAML files so loads fall back to repository defaults."""
    repo = get_repo_config_dir()
    names = {p.name for p in repo.glob("*.yaml")} if repo.is_dir() else set()
    deleted: list[str] = []
    config_dir = get_config_dir()
    for name in sorted(names):
        path = config_dir / name
        if path.is_file():
            path.unlink()
            deleted.append(str(path))
    nested = config_dir / "configs"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()
    startup = config_dir / "startup_profile"
    if startup.is_file():
        startup.unlink()
        deleted.append(str(startup))
    return deleted


def get_startup_profile() -> str | None:
    """Last GUI profile name to load on startup (``~/.config/avlite/startup_profile``)."""
    path = get_config_dir() / "startup_profile"
    if not path.is_file():
        return None
    name = path.read_text(encoding="utf-8").strip()
    return name or None


def set_startup_profile(profile: str) -> None:
    """Remember ``profile`` as the GUI startup profile for the next session."""
    get_config_dir().mkdir(parents=True, exist_ok=True)
    (get_config_dir() / "startup_profile").write_text(profile.strip() + "\n", encoding="utf-8")


def get_plugins_dir() -> Path:
    """Directory where community plugins are installed.

    Honors ``AVLITE_PLUGINS_DIR`` if set, else ``$XDG_DATA_HOME/avlite/plugins``,
    else ``~/.local/share/avlite/plugins``.
    """
    env = os.environ.get("AVLITE_PLUGINS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / DEFAULT_PLUGINS_SUBDIR).resolve()


def installed_community_plugins_map() -> dict[str, str]:
    """Map installed community plugin names to profile storage values."""
    plugins_dir = get_plugins_dir()
    if not plugins_dir.is_dir():
        return {}
    return {
        entry.name: entry.name
        for entry in sorted(plugins_dir.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    }


def resolve_plugin_path(name: str, stored: str) -> Path:
    """Resolve a community plugin install path (name-only, relative, or legacy absolute)."""
    if not stored or stored == name:
        return get_plugins_dir() / name
    path = Path(stored).expanduser()
    if path.is_absolute():
        return path.resolve()
    return get_plugins_dir() / stored


def format_user_path(path: Path | str) -> str:
    """Display a path with ``~`` when it lies under the user home directory."""
    resolved = Path(path).expanduser().resolve()
    try:
        rel = resolved.relative_to(Path.home())
        return "~/" + rel.as_posix()
    except ValueError:
        return resolved.as_posix()


def community_plugin_settings_display_path(name: str) -> str:
    """Human-readable settings file path for a community plugin."""
    return format_user_path(
        effective_config_path(community_plugin_settings_filepath(name), for_write=False)
    )


def plugin_settings_basename(name: str) -> str:
    """YAML basename for a plugin's settings, derived from its directory name.

    Used for both built-in and community plugins so the settings filename is always
    ``plugin_<dir>.yaml`` (e.g. ``p40_bridge_carla`` -> ``plugin_p40_bridge_carla.yaml``).
    """
    return f"plugin_{name}.yaml"


def plugin_settings_filepath(name: str) -> str:
    """Stored filepath token for plugin settings (resolved via ``effective_config_path``)."""
    return f"configs/{plugin_settings_basename(name)}"


# Old, hand-chosen config filenames for built-in plugins, kept only as a read
# fallback so existing user profiles keep loading after the rename to the
# directory-name scheme. New writes always use ``plugin_settings_filepath``.
_LEGACY_PLUGIN_CONFIG: dict[str, str] = {
    "p40_executer_ROS2": "configs/plugin_ros_executer.yaml",
    "p50_headless_mode": "configs/plugin_headless_mode.yaml",
    "p40_bridge_ROS2": "configs/plugin_ROS2_worldbridge.yaml",
    "p30_controller_joystick": "configs/plugin_controller_joystick.yaml",
    "p10_perception_MO_prediction": "configs/plugin_multi_object_predictor.yaml",
    "p40_bridge_gazebo": "configs/plugin_gazebo_worldbridge.yaml",
    "p40_bridge_carla": "configs/plugin_carla.yaml",
}


def legacy_plugin_settings_filepath(name: str) -> str | None:
    """Pre-rename config filepath for a built-in plugin, or ``None`` if unmapped."""
    return _LEGACY_PLUGIN_CONFIG.get(name)


# Backwards-compatible aliases (community plugins use the same scheme).
def community_plugin_settings_basename(name: str) -> str:
    """YAML basename for community plugin profiles under the user config dir."""
    return plugin_settings_basename(name)


def community_plugin_settings_filepath(name: str) -> str:
    """Stored filepath token for community plugin settings (via ``effective_config_path``)."""
    return plugin_settings_filepath(name)


def legacy_community_plugin_settings_path(name: str, stored: str) -> Path:
    """Legacy install-dir settings path (``<install>/config/<name>.yaml``)."""
    return resolve_plugin_path(name, stored) / "config" / f"{name}.yaml"


def normalize_community_plugin_stored(name: str, path_or_stored: str) -> str:
    """Normalize install locator for YAML: name sentinel, ``~/...``, or absolute."""
    if not path_or_stored or path_or_stored == name:
        return name
    path = Path(path_or_stored).expanduser().resolve()
    try:
        path.relative_to(get_plugins_dir())
        return name
    except ValueError:
        pass
    try:
        rel = path.relative_to(Path.home())
        return "~/" + rel.as_posix()
    except ValueError:
        return str(path)


def normalize_community_plugins_map(plugins: dict[str, str]) -> dict[str, str]:
    """Return a copy of *plugins* with portable stored paths."""
    return {
        name: normalize_community_plugin_stored(name, stored)
        for name, stored in plugins.items()
    }


def _legacy_data_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "avlite" / "data").resolve()


def get_data_dir() -> Path:
    """User data directory (~/.config/avlite/data by default)."""
    env = os.environ.get("AVLITE_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (get_config_dir() / "data").resolve()


def get_absolute_path(relative_path: str, *, for_write: bool = False) -> str:
    """Resolve a data path: user dir first on read, repo ``data/`` fallback; writes go to user dir."""
    if os.path.isabs(relative_path):
        if for_write:
            Path(relative_path).parent.mkdir(parents=True, exist_ok=True)
        return relative_path

    repo_root = Path(__file__).resolve().parent.parent.parent
    repo_path = repo_root / relative_path

    rel = relative_path
    if rel.startswith("data/"):
        rel = rel[5:]
    elif rel == "data":
        rel = ""

    if for_write:
        user_path = get_data_dir() / rel
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return str(user_path)

    user_path = get_data_dir() / rel
    if user_path.is_file():
        return str(user_path)
    legacy_path = _legacy_data_dir() / rel
    if legacy_path.is_file():
        return str(legacy_path)
    return str(repo_path)


def builtin_plugins_dir() -> Path:
    """Shipped built-in plugin packages under ``avlite/plugins/``."""
    return Path(__file__).resolve().parent.parent / "plugins"


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def resolve_ui_asset_path(name: str) -> Path:
    """Resolve a UI asset under repo ``data/imgs/`` (independent of process CWD)."""
    path = _repo_data_dir() / "imgs" / name
    if not path.is_file():
        raise FileNotFoundError(f"UI asset not found: {name}")
    return path


def _normalise_geo_degrees(lat: float, lon: float) -> tuple[float, float]:
    """Return WGS84 lat/lon in degrees (OpenDRIVE PROJ may use radians)."""
    if max(abs(lat), abs(lon)) <= math.pi:
        lat, lon = math.degrees(lat), math.degrees(lon)
    return lat, lon


def _parse_proj_lat_lon(proj: str) -> tuple[float, float] | None:
    lat_m = re.search(r"\+lat_0=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", proj)
    lon_m = re.search(r"\+lon_0=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", proj)
    if not lat_m or not lon_m:
        return None
    return _normalise_geo_degrees(float(lat_m.group(1)), float(lon_m.group(1)))


def is_race_boundary_json(abs_path: Path | str) -> bool:
    """True when *abs_path* is a race-boundary JSON with bounds and ReferencePoint."""
    path = Path(abs_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not all(k in data for k in ("LeftBound", "RightBound", "ReferencePoint")):
        return False
    left = data["LeftBound"]
    right = data["RightBound"]
    if not left or not right:
        return False
    if not isinstance(left[0], list) or not isinstance(right[0], list):
        return False
    ref = data["ReferencePoint"]
    if not isinstance(ref, list) or len(ref) < 2:
        return False
    try:
        float(ref[0])
        float(ref[1])
    except (TypeError, ValueError):
        return False
    return True


def is_global_plan_json(abs_path: Path | str) -> bool:
    """True when *abs_path* is a trajectory JSON loadable by ``GlobalPlan.from_file``."""
    path = Path(abs_path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not all(k in data for k in ("ReferenceLine", "ReferenceSpeed", "LeftBound", "RightBound")):
        return False
    ref_line = data["ReferenceLine"]
    ref_speed = data["ReferenceSpeed"]
    left = data["LeftBound"]
    right = data["RightBound"]
    if not ref_line or not ref_speed or not left or not right:
        return False
    if not isinstance(ref_line[0], list) or len(ref_line[0]) < 2:
        return False
    try:
        float(ref_line[0][0])
        float(ref_line[0][1])
    except (TypeError, ValueError, IndexError):
        return False
    if isinstance(left[0], list) or isinstance(right[0], list):
        return False
    try:
        float(left[0])
        float(right[0])
    except (TypeError, ValueError):
        return False
    return True


def extract_reference_point_from_race_json(abs_path: Path | str) -> tuple[float, float] | None:
    path = Path(abs_path)
    if not is_race_boundary_json(path):
        return None
    with path.open(encoding="utf-8") as f:
        ref = json.load(f)["ReferencePoint"]
    return float(ref[0]), float(ref[1])


def extract_reference_point_from_xodr(abs_path: Path | str) -> tuple[float, float] | None:
    path = Path(abs_path)
    if path.suffix.lower() != ".xodr" or not path.is_file():
        return None
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    header = root.find("header")
    if header is None:
        return None
    geo = header.find("geoReference")
    if geo is None or not (geo.text and geo.text.strip()):
        return None
    return _parse_proj_lat_lon(geo.text.strip())


def extract_reference_point_from_map(rel_path: str) -> tuple[float, float] | None:
    """Extract WGS84 (lat_deg, lon_deg) from a repo/user ``data/...`` map path."""
    abs_path = Path(resolve_picker_data_path(rel_path))
    if rel_path.endswith(".xodr"):
        return extract_reference_point_from_xodr(abs_path)
    if is_race_boundary_json(abs_path):
        return extract_reference_point_from_race_json(abs_path)
    return None


def _relative_data_path(file_path: Path, data_root: Path) -> str | None:
    try:
        rel = file_path.relative_to(data_root)
    except ValueError:
        return None
    return "data/" + rel.as_posix()


def format_user_data_picker_path(abs_path: Path) -> str:
    """Format an absolute user-data file for picker display/storage."""
    try:
        rel = abs_path.resolve().relative_to(Path.home())
        return "~/" + rel.as_posix()
    except ValueError:
        return str(abs_path.resolve())


def format_repo_data_picker_path(abs_path: Path) -> str:
    """Format an absolute repo-data file as ``data/...``."""
    rel = abs_path.relative_to(_repo_data_dir())
    return "data/" + rel.as_posix()


def _data_picker_path_for_file(file_path: Path, data_root: Path) -> str:
    if data_root.resolve() == get_data_dir().resolve():
        return format_user_data_picker_path(file_path)
    return format_repo_data_picker_path(file_path)


def resolve_picker_data_path(stored: str, *, for_write: bool = False) -> str:
    """Resolve a picker or settings data path to an absolute filesystem path."""
    if stored.startswith("~/"):
        path = Path(stored).expanduser().resolve()
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return get_absolute_path(stored, for_write=for_write)


def data_picker_path_for_setting(stored: str) -> str:
    """Format a stored settings path for picker display based on resolved location."""
    if stored.startswith("~/"):
        return stored
    abs_path = Path(get_absolute_path(stored)).resolve()
    user_root = get_data_dir().resolve()
    repo_root = _repo_data_dir().resolve()
    try:
        abs_path.relative_to(user_root)
        return format_user_data_picker_path(abs_path)
    except ValueError:
        pass
    try:
        rel = abs_path.relative_to(repo_root)
        return "data/" + rel.as_posix()
    except ValueError:
        return stored


def _iter_data_files(*roots: Path):
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                yield path, root


def _collect_data_picker_candidates(predicate) -> list[str]:
    """Return sorted picker paths (user dir first, then repo) passing *predicate*."""
    repo_data = _repo_data_dir()
    user_data = get_data_dir()
    seen: set[str] = set()
    user_candidates: list[str] = []
    repo_candidates: list[str] = []

    for path, root in _iter_data_files(user_data, repo_data):
        if not predicate(path):
            continue
        picker_path = _data_picker_path_for_file(path, root)
        if picker_path in seen:
            continue
        seen.add(picker_path)
        if root.resolve() == user_data.resolve():
            user_candidates.append(picker_path)
        else:
            repo_candidates.append(picker_path)

    return sorted(user_candidates) + sorted(repo_candidates)


def list_map_file_candidates() -> list[str]:
    """Sorted picker paths for OpenDRIVE maps and race-boundary JSON files."""
    def _is_map(path: Path) -> bool:
        if path.suffix.lower() == ".xodr":
            return True
        return is_race_boundary_json(path)

    return _collect_data_picker_candidates(_is_map)


def list_global_plan_file_candidates() -> list[str]:
    """Sorted picker paths for global-plan trajectory JSON files."""
    return _collect_data_picker_candidates(is_global_plan_json)


def apply_map_selection(rel_path: str) -> None:
    """Route *rel_path* to execution map settings and update reference point."""
    from avlite.c40_execution.c49_settings import ExecutionSettings

    abs_path = resolve_picker_data_path(rel_path)
    if rel_path.endswith(".xodr"):
        ExecutionSettings.c40_hd_map = rel_path
        ExecutionSettings.c46_lidar_boundary_file = ""
    elif is_race_boundary_json(abs_path):
        ExecutionSettings.c43_race_boundary_map = rel_path
        ExecutionSettings.c46_lidar_boundary_file = rel_path
    ref = extract_reference_point_from_map(rel_path)
    ExecutionSettings.c40_reference_point = list(ref) if ref else None


def apply_global_plan_selection(rel_path: str) -> None:
    """Set ``c40_global_trajectory`` when *rel_path* is a valid global-plan JSON."""
    from avlite.c40_execution.c49_settings import ExecutionSettings

    if is_global_plan_json(resolve_picker_data_path(rel_path)):
        ExecutionSettings.c40_global_trajectory = rel_path


def bootstrap_reference_point_from_maps() -> None:
    """Fill ``c40_reference_point`` from configured maps when YAML omits it."""
    from avlite.c40_execution.c49_settings import ExecutionSettings

    if ExecutionSettings.c40_reference_point is not None:
        return
    for rel_path in (
        ExecutionSettings.c43_race_boundary_map,
        ExecutionSettings.c40_hd_map,
    ):
        ref = extract_reference_point_from_map(rel_path)
        if ref is not None:
            ExecutionSettings.c40_reference_point = list(ref)
            return
