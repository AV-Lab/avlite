"""Filesystem paths for AVLite configs, data, and community plugins."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_PLUGINS_SUBDIR = Path("avlite") / "plugins"


def get_config_dir() -> Path:
    """User AVLite config root (~/.config/avlite by default)."""
    env = os.environ.get("AVLITE_CONFIG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "avlite").resolve()


def _user_meta_dir() -> Path:
    """Preferences always under ~/.config/avlite (ignores ``AVLITE_CONFIG_DIR``)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "avlite").resolve()


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


def copy_repo_configs_to_user() -> list[str]:
    """Copy bundled ``configs/*.yaml`` into ``get_config_dir()``; overwrite existing."""
    src_dir = bundled_config_dir()
    if not can_edit_repo_configs():
        raise FileNotFoundError(f"No bundled configs in {src_dir}")
    dest_dir = get_config_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(src_dir.glob("*.yaml")):
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied.append(str(dest))
    return copied


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
    path = Path(stored)
    if path.is_absolute():
        return path
    return get_plugins_dir() / stored


def get_data_dir() -> Path:
    """User data directory (~/.local/share/avlite/data by default)."""
    env = os.environ.get("AVLITE_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "avlite" / "data").resolve()


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
    return str(repo_path)


def builtin_plugins_dir() -> Path:
    """Shipped built-in plugin packages under ``avlite/plugins/``."""
    return Path(__file__).resolve().parent.parent / "plugins"
