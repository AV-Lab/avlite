"""Community plugins browser/installer app for AVLite.

Browses the AV-Lab community plugin registry, installs/uninstalls plugins
into a user-data directory, and (de)registers them with the active
execution profile.

Standalone:  ``avlite plugins`` (or ``python -m avlite plugins``)
Embedded:    ``CommunityPluginsApp.open(parent)`` from the main app.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import yaml

log = logging.getLogger(__name__)

REGISTRY_URL = (
    "https://raw.githubusercontent.com/AV-Lab/avlite-community-plugins/main/plugins.yaml"
)
REGISTRY_REPO_URL = "https://github.com/AV-Lab/avlite-community-plugins"
DEFAULT_PLUGINS_SUBDIR = Path("avlite") / "plugins"


# ---------------------------------------------------------------------------
# Core (Tk-free) helpers
# ---------------------------------------------------------------------------
def get_plugins_dir() -> Path:
    """Return the directory where community plugins are installed.

    Honors ``AVLITE_PLUGINS_DIR`` if set, else ``$XDG_DATA_HOME/avlite/plugins``,
    else ``~/.local/share/avlite/plugins``.
    """
    env = os.environ.get("AVLITE_PLUGINS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / DEFAULT_PLUGINS_SUBDIR).resolve()


def fetch_registry(timeout: float = 10.0) -> list[dict]:
    """Fetch and parse the community plugins registry."""
    req = urllib.request.Request(REGISTRY_URL, headers={"User-Agent": "avlite"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = yaml.safe_load(resp.read()) or {}
    plugins = data.get("plugins") or []
    if not isinstance(plugins, list):
        raise ValueError("Registry plugins.yaml has unexpected schema")
    return plugins


def list_installed(plugins_dir: Path) -> list[dict]:
    """List plugin directories present under ``plugins_dir``."""
    out: list[dict] = []
    if not plugins_dir.exists():
        return out
    for entry in sorted(plugins_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        out.append({
            "name": entry.name,
            "path": entry,
            "has_init": (entry / "__init__.py").exists(),
        })
    return out


def install_plugin(entry: dict, plugins_dir: Path) -> Path:
    """Clone the plugin repo into ``plugins_dir`` and checkout the version.

    Returns the absolute install path.
    """
    name = entry["name"]
    repo = entry["repository"]
    version = entry.get("version", "latest")
    plugins_dir.mkdir(parents=True, exist_ok=True)
    target = (plugins_dir / name).resolve()

    if target.exists():
        raise FileExistsError(f"Plugin '{name}' already installed at {target}")

    # Safety: target must remain inside plugins_dir
    if plugins_dir.resolve() not in target.parents:
        raise ValueError(f"Refusing to install outside plugins dir: {target}")

    log.info("Cloning %s -> %s", repo, target)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo, str(target)]
        if version == "latest"
        else ["git", "clone", repo, str(target)],
        check=True,
        capture_output=True,
        text=True,
    )

    if version and version != "latest":
        log.info("Checking out version %s", version)
        subprocess.run(
            ["git", "-C", str(target), "checkout", version],
            check=True,
            capture_output=True,
            text=True,
        )
    return target


def uninstall_plugin(name: str, plugins_dir: Path) -> None:
    """Remove an installed plugin directory (guarded to plugins_dir)."""
    plugins_dir = plugins_dir.resolve()
    target = (plugins_dir / name).resolve()
    if plugins_dir not in target.parents:
        raise ValueError(f"Refusing to delete outside plugins dir: {target}")
    if not target.exists():
        log.warning("Plugin directory not found: %s", target)
        return
    log.info("Removing %s", target)
    shutil.rmtree(target)


def check_requirements(req_file: Path) -> tuple[list[str], list[str]]:
    """Inspect ``requirements.txt`` vs current env. Returns (missing, mismatched)."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except Exception:
        return [], []

    missing: list[str] = []
    mismatched: list[str] = []
    for raw in req_file.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        try:
            installed = pkg_version(req.name)
        except PackageNotFoundError:
            missing.append(line)
            continue
        if req.specifier and not req.specifier.contains(Version(installed), prereleases=True):
            mismatched.append(f"{line} (installed {installed})")
    return missing, mismatched


def pip_install(req_file: Path) -> None:
    """Install requirements from ``req_file`` into the current interpreter."""
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        check=True,
        capture_output=True,
        text=True,
    )


def get_local_head(plugin_path: Path) -> Optional[str]:
    """Return the current HEAD commit SHA of a local git repo, or ``None``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(plugin_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_remote_sha(repo_url: str, ref: str = "HEAD") -> Optional[str]:
    """Return the commit SHA for *ref* on the remote, or ``None`` on failure.

    Uses ``ref^{}`` to dereference annotated tags to the underlying commit SHA.
    The last matching line from ``git ls-remote`` is used.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, ref, f"{ref}^{{}}"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return None
        sha = None
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if parts:
                sha = parts[0]
        return sha
    except Exception:
        return None


def check_plugin_update(plugin_path: Path, registry_entry: Optional[dict]) -> str:
    """Compare local HEAD against remote ref.

    Returns ``'up-to-date'``, ``'update-available'``, or ``'unknown'``.
    """
    local_sha = get_local_head(plugin_path)
    if local_sha is None:
        return "unknown"
    if registry_entry is not None:
        version = registry_entry.get("version", "latest")
        repo = registry_entry.get("repository", "")
    else:
        try:
            r = subprocess.run(
                ["git", "-C", str(plugin_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            repo = r.stdout.strip() if r.returncode == 0 else ""
            version = "latest"
        except Exception:
            return "unknown"
    if not repo:
        return "unknown"
    ref = "HEAD" if version == "latest" else f"refs/tags/{version}"
    remote_sha = get_remote_sha(repo, ref)
    if remote_sha is None:
        return "unknown"
    return "up-to-date" if local_sha == remote_sha else "update-available"


def update_plugin(plugin_path: Path, version: str = "latest") -> None:
    """Pull the latest commit, or fetch and check out a specific tag."""
    if version == "latest":
        subprocess.run(
            ["git", "-C", str(plugin_path), "pull"],
            check=True, capture_output=True, text=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(plugin_path), "fetch", "origin"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(plugin_path), "checkout", version],
            check=True, capture_output=True, text=True,
        )


def _current_profile() -> str:
    """Best-effort: read the active profile from the visualization settings file."""
    try:
        from avlite.c50_visualization.c59_settings import VisualizationSettings

        path = Path(VisualizationSettings.filepath)
        if path.exists():
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            for prof, body in cfg.items():
                if isinstance(body, dict) and body.get("selected_profile") == prof:
                    return prof
            # Fallback: first profile in file
            if cfg:
                return next(iter(cfg.keys()))
    except Exception as e:
        log.debug("Could not determine active profile: %s", e)
    return "default"


def register_in_profile(name: str, path: Path, profile: Optional[str] = None) -> None:
    """Add ``name -> path`` to ``ExecutionSettings.c40_community_plugins`` and persist."""
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_common.c61_setting_utils import load_setting, save_setting

    profile = profile or _current_profile()
    # Load existing profile state so we don't overwrite unrelated entries.
    load_setting(ExecutionSettings, profile=profile)
    ExecutionSettings.c40_community_plugins[name] = str(path)
    save_setting(ExecutionSettings, profile=profile)
    log.info("Registered plugin '%s' in profile '%s'", name, profile)


def unregister_from_profile(name: str, profile: Optional[str] = None) -> None:
    """Remove ``name`` from ``ExecutionSettings.c40_community_plugins`` and persist."""
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_common.c61_setting_utils import load_setting, save_setting

    profile = profile or _current_profile()
    load_setting(ExecutionSettings, profile=profile)
    ExecutionSettings.c40_community_plugins.pop(name, None)
    save_setting(ExecutionSettings, profile=profile)
    log.info("Unregistered plugin '%s' from profile '%s'", name, profile)


def _registered_names() -> set[str]:
    try:
        from avlite.c40_execution.c49_settings import ExecutionSettings

        return set(ExecutionSettings.c40_community_plugins.keys())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class CommunityPluginsApp:
    """Standalone window for browsing/installing community plugins."""

    _instance: "Optional[CommunityPluginsApp]" = None

    COLUMNS = ("name", "category", "version", "status", "update_status", "path")

    def __init__(self, parent: Optional[tk.Misc] = None):
        self.parent = parent
        self.plugins_dir = get_plugins_dir()
        self._busy = False
        self._registry: list[dict] = []
        self._update_statuses: dict[str, str] = {}
        self._owns_root = parent is None

        if parent is None:
            self.window: tk.Misc = tk.Tk()
        else:
            self.window = tk.Toplevel(parent)
            try:
                self.window.transient(parent)
            except tk.TclError:
                pass

        _s = max(1.0, min(3.0, self.window.winfo_fpixels('1i') / 96.0))
        self.window.title("AVLite Community Plugins")
        self.window.geometry(f"{round(900 * _s)}x{round(500 * _s)}")
        self.window.minsize(round(700 * _s), round(350 * _s))
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.bind("<Escape>", lambda _e: self._on_close())

        # Match window background to the ttk theme so no border shows around frames.
        try:
            bg = ttk.Style(self.window).lookup("TFrame", "background")
            if bg:
                self.window.configure(background=bg)
        except tk.TclError:
            pass

        self._build_ui()
        self._refresh_async()

    # -- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.window)
        outer.pack(fill=tk.BOTH, expand=True)
        outer = ttk.Frame(outer, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            header,
            text=f"Plugins directory: {self.plugins_dir}",
            foreground="#666",
        ).pack(side=tk.LEFT)

        # Tree
        tree_frame = ttk.Frame(outer)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            tree_frame, columns=self.COLUMNS, show="headings", selectmode="browse"
        )
        headings = {
            "name": ("Name", 180),
            "category": ("Category", 110),
            "version": ("Version", 80),
            "status": ("Status", 130),
            "update_status": ("Update", 120),
            "path": ("Repository / Path", 300),
        }
        for col, (label, width) in headings.items():
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor=tk.W, stretch=(col == "path"))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._update_buttons())
        self.tree.bind("<Double-Button-1>", lambda _e: self._on_default_action())

        # Toolbar
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(8, 0))

        self.btn_refresh = ttk.Button(toolbar, text="Refresh", command=self._refresh_async)
        self.btn_install = ttk.Button(toolbar, text="Install", command=self._on_install)
        self.btn_uninstall = ttk.Button(toolbar, text="Uninstall", command=self._on_uninstall)
        self.btn_update = ttk.Button(toolbar, text="Update", command=self._on_update)
        self.btn_update_all = ttk.Button(toolbar, text="Update All", command=self._on_update_all)
        self.btn_open = ttk.Button(toolbar, text="Open Folder", command=self._open_folder)
        self.btn_close = ttk.Button(toolbar, text="Close", command=self._on_close)

        for b in (self.btn_refresh, self.btn_install, self.btn_uninstall,
                  self.btn_update, self.btn_update_all, self.btn_open):
            b.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_close.pack(side=tk.RIGHT)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status_var, anchor=tk.W).pack(
            fill=tk.X, pady=(6, 0)
        )

        self._update_buttons()

    # -- Population ------------------------------------------------------
    def _populate(self) -> None:
        self.tree.delete(*self.tree.get_children())
        installed = {p["name"]: p for p in list_installed(self.plugins_dir)}
        registry_by_name = {e["name"]: e for e in self._registry}
        registered = _registered_names()

        # Registry entries (Available or Installed)
        for entry in self._registry:
            name = entry["name"]
            inst = installed.pop(name, None)
            if inst is not None:
                status = "Installed"
                if name in registered:
                    status += " ✓"
                path = str(inst["path"])
            else:
                status = "Available"
                path = entry.get("repository", "")
            up_st = self._update_statuses.get(name, "Checking…") if inst is not None else ""
            self.tree.insert(
                "",
                tk.END,
                iid=name,
                values=(
                    name,
                    entry.get("category", ""),
                    entry.get("version", ""),
                    status,
                    up_st,
                    path,
                ),
            )

        # Installed but not in registry (local)
        for name, inst in sorted(installed.items()):
            status = "Installed (local)"
            if name in registered:
                status += " ✓"
            up_st = self._update_statuses.get(name, "Checking…")
            self.tree.insert(
                "",
                tk.END,
                iid=name,
                values=(name, "", "", status, up_st, str(inst["path"])),
            )
        self._check_updates_async()
        self._update_buttons()

    def _selected_entry(self) -> Optional[tuple[str, str]]:
        sel = self.tree.selection()
        if not sel:
            return None
        name = sel[0]
        status = self.tree.set(name, "status")
        return name, status

    def _update_buttons(self) -> None:
        sel = self._selected_entry()
        installed = sel is not None and sel[1].startswith("Installed")
        available = sel is not None and sel[1] == "Available"
        busy = self._busy
        has_update = (
            installed and sel is not None
            and self._update_statuses.get(sel[0]) == "update-available"
        )
        has_any_update = any(s == "update-available" for s in self._update_statuses.values())
        self.btn_install.state(["!disabled"] if (available and not busy) else ["disabled"])
        self.btn_uninstall.state(["!disabled"] if (installed and not busy) else ["disabled"])
        self.btn_refresh.state(["disabled"] if busy else ["!disabled"])
        self.btn_update.state(["!disabled"] if (has_update and not busy) else ["disabled"])
        self.btn_update_all.state(["!disabled"] if (has_any_update and not busy) else ["disabled"])

    def _on_default_action(self) -> None:
        sel = self._selected_entry()
        if not sel:
            return
        if sel[1] == "Available":
            self._on_install()
        elif sel[1].startswith("Installed"):
            self._on_uninstall()

    # -- Actions ---------------------------------------------------------
    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self._busy = busy
        if msg:
            self.status_var.set(msg)
        self._update_buttons()

    def _run_bg(self, fn, on_done) -> None:
        def worker():
            try:
                result = fn()
                self.window.after(0, lambda: on_done(result, None))
            except Exception as exc:  # noqa: BLE001
                log.exception("Background task failed")
                self.window.after(0, lambda: on_done(None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_async(self) -> None:
        self._set_busy(True, "Fetching registry…")

        def done(result, err):
            if err is not None:
                self._registry = []
                self._populate()
                self._set_busy(False, f"Failed to load registry: {err}")
                return
            self._registry = result or []
            self._update_statuses.clear()
            self._populate()
            self._set_busy(False, f"Loaded {len(self._registry)} plugin(s) from registry.")

        self._run_bg(fetch_registry, done)

    def _check_updates_async(self) -> None:
        """Start background update checks for installed plugins not yet checked."""
        installed = {p["name"]: p for p in list_installed(self.plugins_dir)}
        registry_by_name = {e["name"]: e for e in self._registry}
        names_to_check = [n for n in installed if n not in self._update_statuses]
        if not names_to_check:
            return

        def run_all():
            for name in names_to_check:
                plugin_path = installed[name]["path"]
                registry_entry = registry_by_name.get(name)
                try:
                    result = check_plugin_update(plugin_path, registry_entry)
                except Exception as e:
                    log.debug("Update check failed for %s: %s", name, e)
                    result = "unknown"
                self.window.after(0, lambda n=name, r=result: self._on_update_check_done(n, r))

        threading.Thread(target=run_all, daemon=True).start()

    def _on_update_check_done(self, name: str, result: str) -> None:
        """Called on the main thread when a single plugin's update check finishes."""
        self._update_statuses[name] = result
        label_map = {
            "up-to-date": "Up to date \u2713",
            "update-available": "Update \u2191",
            "unknown": "\u2014",
        }
        label = label_map.get(result, "\u2014")
        try:
            if self.tree.exists(name):
                self.tree.set(name, "update_status", label)
        except tk.TclError:
            pass
        self._update_buttons()

    def _on_update(self) -> None:
        """Update the currently selected plugin."""
        sel = self._selected_entry()
        if not sel or not sel[1].startswith("Installed"):
            return
        name = sel[0]
        if self._update_statuses.get(name) != "update-available":
            return
        self._update_single(name)

    def _update_single(self, name: str) -> None:
        installed = {p["name"]: p for p in list_installed(self.plugins_dir)}
        plugin_path = installed.get(name, {}).get("path")
        if plugin_path is None:
            return
        registry_entry = next((e for e in self._registry if e["name"] == name), None)
        version = registry_entry.get("version", "latest") if registry_entry else "latest"
        self._set_busy(True, f"Updating {name}\u2026")

        def task():
            update_plugin(plugin_path, version)

        def done(_result, err):
            if err is not None:
                self._set_busy(False, f"Update failed for {name}: {err}")
                messagebox.showerror("Update failed", str(err), parent=self.window)
                return
            self._update_statuses.pop(name, None)
            self._set_busy(False, f"Updated {name}.")
            self._populate()
            self._notify_host_changed()

        self._run_bg(task, done)

    def _on_update_all(self) -> None:
        """Update all plugins that have an available update."""
        names = [n for n, s in self._update_statuses.items() if s == "update-available"]
        if not names:
            return
        installed_map = {p["name"]: p for p in list_installed(self.plugins_dir)}
        registry_by_name = {e["name"]: e for e in self._registry}
        self._set_busy(True, f"Updating {len(names)} plugin(s)\u2026")

        def task():
            errors: list[str] = []
            for name in names:
                path = installed_map.get(name, {}).get("path")
                if path is None:
                    continue
                entry = registry_by_name.get(name)
                version = entry.get("version", "latest") if entry else "latest"
                try:
                    update_plugin(path, version)
                    self._update_statuses.pop(name, None)
                except Exception as e:
                    errors.append(f"{name}: {e}")
            return errors

        def done(errors, err):
            if err is not None:
                self._set_busy(False, f"Update all failed: {err}")
                return
            if errors:
                messagebox.showwarning(
                    "Some updates failed", "\n".join(errors), parent=self.window
                )
            updated = len(names) - len(errors or [])
            self._set_busy(False, f"Updated {updated} plugin(s).")
            self._populate()
            self._notify_host_changed()

        self._run_bg(task, done)

    def _active_profile(self) -> Optional[str]:
        host = self.parent
        if host is None:
            return None
        try:
            return host.setting.selected_profile.get()
        except Exception:
            return None

    def _on_install(self) -> None:
        sel = self._selected_entry()
        if not sel or sel[1] != "Available":
            return
        name = sel[0]
        entry = next((e for e in self._registry if e["name"] == name), None)
        if entry is None:
            return
        profile = self._active_profile()
        self._set_busy(True, f"Installing {name}…")

        def task():
            path = install_plugin(entry, self.plugins_dir)
            register_in_profile(name, path, profile=profile)
            return path

        def done(path, err):
            if err is not None:
                self._set_busy(False, f"Install failed: {err}")
                messagebox.showerror("Install failed", str(err), parent=self.window)
                return
            self._handle_requirements(name, path)
            self._set_busy(False, f"Installed {name} at {path}")
            self._populate()
            self._notify_host_changed()

        self._run_bg(task, done)

    def _handle_requirements(self, name: str, plugin_path: Path) -> None:
        """Check the plugin's requirements.txt and prompt to install missing deps."""
        req_file = plugin_path / "requirements.txt"
        if not req_file.exists():
            return
        missing, mismatched = check_requirements(req_file)
        if mismatched:
            messagebox.showwarning(
                "Dependency version mismatch",
                f"'{name}' requires:\n  " + "\n  ".join(mismatched)
                + "\n\nThe plugin may not work correctly.",
                parent=self.window,
            )
        if missing and messagebox.askyesno(
            "Install missing dependencies?",
            f"'{name}' needs:\n  " + "\n  ".join(missing)
            + "\n\nInstall them into the current Python environment?",
            parent=self.window,
        ):
            try:
                pip_install(req_file)
            except subprocess.CalledProcessError as e:
                detail = "\n".join(p for p in (e.stdout, e.stderr) if p) or str(e)
                messagebox.showerror(
                    "pip install failed",
                    detail,
                    parent=self.window,
                )

    def _on_uninstall(self) -> None:
        sel = self._selected_entry()
        if not sel or not sel[1].startswith("Installed"):
            return
        name = sel[0]
        if not messagebox.askyesno(
            "Uninstall plugin",
            f"Remove '{name}' from the profile and delete its files?",
            parent=self.window,
        ):
            return
        profile = self._active_profile()
        self._set_busy(True, f"Uninstalling {name}…")

        def task():
            unregister_from_profile(name, profile=profile)
            uninstall_plugin(name, self.plugins_dir)

        def done(_result, err):
            if err is not None:
                self._set_busy(False, f"Uninstall failed: {err}")
                messagebox.showerror("Uninstall failed", str(err), parent=self.window)
                return
            self._set_busy(False, f"Uninstalled {name}")
            self._populate()
            self._notify_host_changed()

        self._run_bg(task, done)

    def _open_folder(self) -> None:
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(self.plugins_dir))  # type: ignore[attr-defined]
            elif "darwin" in os.sys.platform:  # type: ignore[attr-defined]
                subprocess.Popen(["open", str(self.plugins_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.plugins_dir)])
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Could not open folder: {e}")

    def _notify_host_changed(self) -> None:
        """Best-effort hook so an embedding visualizer can refresh its UI."""
        host = self.parent
        if host is None:
            return
        try:
            cfg_view = getattr(host, "config_shortcut_view", None)
            if cfg_view is not None and hasattr(cfg_view, "update_setting_window"):
                cfg_view.update_setting_window()
        except Exception:
            log.debug("Host refresh hook failed", exc_info=True)

    # -- Lifecycle -------------------------------------------------------
    def _on_close(self) -> None:
        CommunityPluginsApp._instance = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    # -- Public factory --------------------------------------------------
    @classmethod
    def open(cls, parent: Optional[tk.Misc] = None) -> "CommunityPluginsApp":
        existing = cls._instance
        if existing is not None:
            try:
                existing.window.deiconify()
                existing.window.lift()
                existing.window.focus_force()
                return existing
            except tk.TclError:
                cls._instance = None

        app = cls(parent)
        cls._instance = app
        if parent is None:
            app.window.mainloop()
        return app


def main() -> None:
    """Entry point for ``avlite plugins``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    CommunityPluginsApp.open(parent=None)


if __name__ == "__main__":
    main()
