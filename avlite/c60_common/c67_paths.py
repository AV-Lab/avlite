"""Filesystem paths for AVLite configs, data, and community plugins."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PLUGINS_SUBDIR = Path("avlite") / "plugins"


class ConfigPaths:
    """User and repository configuration directory resolution."""

    @staticmethod
    def user_dir() -> Path:
        """User AVLite config root (~/.config/avlite by default)."""
        env = os.environ.get("AVLITE_CONFIG_DIR", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return (ConfigPaths._xdg_config_base() / "avlite").resolve()

    @staticmethod
    def user_configs_dir() -> Path:
        """User YAML profiles directory (same as ``user_dir()``)."""
        return ConfigPaths.user_dir()

    @staticmethod
    def bundled_dir() -> Path:
        """Shipped YAML directory (git clone ``configs/``); missing after plain pip install."""
        return Path(__file__).resolve().parent.parent.parent / "configs"

    @staticmethod
    def can_edit_bundled() -> bool:
        d = ConfigPaths.bundled_dir()
        return d.is_dir() and any(d.glob("*.yaml"))

    @staticmethod
    def is_repo_target() -> bool:
        path = ConfigPaths._user_meta_dir() / "config_target"
        if not path.is_file():
            return False
        return path.read_text(encoding="utf-8").strip() == "repo"

    @staticmethod
    def set_repo_target(enabled: bool) -> None:
        ConfigPaths._user_meta_dir().mkdir(parents=True, exist_ok=True)
        value = "repo" if enabled else "user"
        (ConfigPaths._user_meta_dir() / "config_target").write_text(value + "\n", encoding="utf-8")

    @staticmethod
    def effective_path(stored_filepath: str, *, for_write: bool = False) -> str:
        """Resolve a settings filepath: user config dir for writes; user if present else repo for reads."""
        path = Path(stored_filepath)
        if path.is_absolute():
            if for_write:
                path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        name = path.name
        user = ConfigPaths.user_dir() / name
        repo = ConfigPaths.bundled_dir() / name
        if ConfigPaths.is_repo_target() and ConfigPaths.can_edit_bundled():
            bundled = ConfigPaths.bundled_dir() / name
            if for_write:
                bundled.parent.mkdir(parents=True, exist_ok=True)
            return str(bundled)
        if for_write:
            ConfigPaths.user_dir().mkdir(parents=True, exist_ok=True)
            return str(user)
        return str(user if user.is_file() else repo)

    @staticmethod
    def clear_user_profiles() -> list[str]:
        """Remove user-local stack YAML files so loads fall back to repository defaults."""
        repo = ConfigPaths.bundled_dir()
        names = {p.name for p in repo.glob("*.yaml")} if repo.is_dir() else set()
        deleted: list[str] = []
        config_dir = ConfigPaths.user_dir()
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

    @staticmethod
    def startup_profile() -> str | None:
        """Last GUI profile name to load on startup."""
        path = ConfigPaths.user_dir() / "startup_profile"
        if not path.is_file():
            return None
        name = path.read_text(encoding="utf-8").strip()
        return name or None

    @staticmethod
    def set_startup_profile(profile: str) -> None:
        """Remember *profile* as the GUI startup profile for the next session."""
        ConfigPaths.user_dir().mkdir(parents=True, exist_ok=True)
        (ConfigPaths.user_dir() / "startup_profile").write_text(profile.strip() + "\n", encoding="utf-8")

    @staticmethod
    def _xdg_config_base() -> Path:
        xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
        return Path(xdg).expanduser() if xdg else Path.home() / ".config"

    @staticmethod
    def _user_meta_dir() -> Path:
        """Preferences always under ~/.config/avlite (ignores ``AVLITE_CONFIG_DIR``)."""
        return (ConfigPaths._xdg_config_base() / "avlite").resolve()


class PluginPaths:
    """Built-in and community plugin install and settings paths."""

    @staticmethod
    def install_dir() -> Path:
        """Directory where community plugins are installed."""
        env = os.environ.get("AVLITE_PLUGINS_DIR")
        if env:
            return Path(env).expanduser().resolve()
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        return (base / DEFAULT_PLUGINS_SUBDIR).resolve()

    @staticmethod
    def installed_map() -> dict[str, str]:
        """Map installed community plugin names to profile storage values."""
        plugins_dir = PluginPaths.install_dir()
        if not plugins_dir.is_dir():
            return {}
        return {
            entry.name: entry.name
            for entry in sorted(plugins_dir.iterdir())
            if entry.is_dir() and not entry.name.startswith(".")
        }

    @staticmethod
    def repo_root() -> Path:
        """AVLite repository root (parent of ``avlite/`` package)."""
        return Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def resolve(name: str, stored: str) -> Path:
        """Resolve a community plugin install path (name-only, relative, or legacy absolute)."""
        if not stored or stored == name:
            return PluginPaths.install_dir() / name
        path = Path(stored).expanduser()
        if path.is_absolute():
            return path.resolve()
        repo_relative = PluginPaths.repo_root() / stored
        if repo_relative.is_dir():
            return repo_relative.resolve()
        return PluginPaths.install_dir() / stored

    @staticmethod
    def format_display(path: Path | str) -> str:
        """Display a path with ``~`` when it lies under the user home directory."""
        resolved = Path(path).expanduser().resolve()
        try:
            rel = resolved.relative_to(Path.home())
            return "~/" + rel.as_posix()
        except ValueError:
            return resolved.as_posix()

    @staticmethod
    def settings_display_path(name: str) -> str:
        """Human-readable settings file path for a community plugin."""
        return PluginPaths.format_display(
            ConfigPaths.effective_path(PluginPaths.settings_filepath(name), for_write=False)
        )

    @staticmethod
    def settings_basename(name: str) -> str:
        """YAML basename for a plugin's settings (``plugin_<dir>.yaml``)."""
        return f"plugin_{name}.yaml"

    @staticmethod
    def settings_filepath(name: str) -> str:
        """Stored filepath token for plugin settings."""
        return f"configs/{PluginPaths.settings_basename(name)}"

    @staticmethod
    def normalize_stored(name: str, path_or_stored: str) -> str:
        """Normalize install locator for YAML: name sentinel, ``~/...``, or absolute."""
        if not path_or_stored or path_or_stored == name:
            return name
        path = Path(path_or_stored).expanduser().resolve()
        try:
            path.relative_to(PluginPaths.install_dir())
            return name
        except ValueError:
            pass
        try:
            rel = path.relative_to(PluginPaths.repo_root())
            return rel.as_posix()
        except ValueError:
            pass
        try:
            rel = path.relative_to(Path.home())
            return "~/" + rel.as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def normalize_map(plugins: dict[str, str]) -> dict[str, str]:
        """Return a copy of *plugins* with portable stored paths."""
        return {
            name: PluginPaths.normalize_stored(name, stored)
            for name, stored in plugins.items()
        }

    @staticmethod
    def builtin_dir() -> Path:
        """Shipped built-in plugin packages under ``avlite/plugins/``."""
        return Path(__file__).resolve().parent.parent / "plugins"


class DataPaths:
    """User and repository data file resolution."""

    @staticmethod
    def user_dir() -> Path:
        """User data directory (~/.config/avlite/data by default)."""
        env = os.environ.get("AVLITE_DATA_DIR", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return (ConfigPaths.user_dir() / "data").resolve()

    @staticmethod
    def resolve(relative_path: str, *, for_write: bool = False) -> str:
        """Resolve a data path: user dir first on read, repo ``data/`` fallback."""
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
            user_path = DataPaths.user_dir() / rel
            user_path.parent.mkdir(parents=True, exist_ok=True)
            return str(user_path)

        user_path = DataPaths.user_dir() / rel
        if user_path.is_file():
            return str(user_path)

        def _legacy_data_dir() -> Path:
            xdg = os.environ.get("XDG_DATA_HOME", "").strip()
            base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
            return (base / "avlite" / "data").resolve()

        legacy_path = _legacy_data_dir() / rel
        if legacy_path.is_file():
            return str(legacy_path)
        return str(repo_path)

    @staticmethod
    def resolve_stored(stored: str, *, for_write: bool = False) -> str:
        """Resolve a picker or settings data path to an absolute filesystem path."""
        if stored.startswith("~/"):
            path = Path(stored).expanduser().resolve()
            if for_write:
                path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        return DataPaths.resolve(stored, for_write=for_write)

    @staticmethod
    def _repo_data_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent / "data"
