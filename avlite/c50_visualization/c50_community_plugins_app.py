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
import re
import shutil
import subprocess
import threading
import tkinter as tk
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

import yaml

from avlite.c50_visualization.c58_ui_lib import (
    configure_treeview_style,
    get_dpi_scale,
    scaled,
    scaled_font,
)

log = logging.getLogger(__name__)

REGISTRY_URL = (
    "https://raw.githubusercontent.com/AV-Lab/avlite-community-plugins/main/plugins.yaml"
)
REGISTRY_REPO_URL = "https://github.com/AV-Lab/avlite-community-plugins"
from avlite.c60_common.c67_paths import get_plugins_dir


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
    from avlite.c60_common.c69_setting_utils import load_setting, save_setting

    profile = profile or _current_profile()
    # Load existing profile state so we don't overwrite unrelated entries.
    load_setting(ExecutionSettings, profile=profile)
    path = path.resolve()
    try:
        path.relative_to(get_plugins_dir())
        stored = name
    except ValueError:
        stored = str(path)
    ExecutionSettings.c40_community_plugins[name] = stored
    save_setting(ExecutionSettings, profile=profile)
    log.info("Registered plugin '%s' in profile '%s'", name, profile)


def unregister_from_profile(name: str, profile: Optional[str] = None) -> None:
    """Remove ``name`` from ``ExecutionSettings.c40_community_plugins`` and persist."""
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_common.c69_setting_utils import load_setting, save_setting

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


def _parse_github_repo(repository: str) -> Optional[tuple[str, str]]:
    repo = repository.rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    for prefix in ("https://github.com/", "http://github.com/"):
        if repo.startswith(prefix):
            parts = repo[len(prefix):].split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
    return None


def _normalize_repo_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@"):
        host_path = url[4:]
        if ":" in host_path:
            host, path = host_path.split(":", 1)
            return f"https://{host}/{path}"
    return url


def get_plugin_repository_url(
    registry_entry: Optional[dict],
    install_path: Optional[Path],
) -> Optional[str]:
    """Return a browser-friendly GitHub URL for a plugin, if known."""
    if registry_entry:
        repo = registry_entry.get("repository", "")
        if repo:
            return _normalize_repo_url(repo)
    if install_path is not None:
        try:
            result = subprocess.run(
                ["git", "-C", str(install_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return _normalize_repo_url(result.stdout.strip())
        except Exception:
            pass
    return None


def read_local_readme(plugin_path: Path) -> str:
    """Read README from an installed plugin directory."""
    for name in ("README.md", "readme.md", "Readme.md"):
        path = plugin_path / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def fetch_remote_readme(repository: str, version: str = "latest", timeout: float = 10.0) -> str:
    """Fetch README.md from a GitHub repository via raw.githubusercontent.com."""
    parsed = _parse_github_repo(repository)
    if parsed is None:
        return ""
    owner, repo = parsed
    refs = [version] if version and version != "latest" else ["main", "master", "HEAD"]
    for ref in refs:
        for readme_name in ("README.md", "readme.md"):
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{readme_name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "avlite"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except Exception:
                continue
    return ""


def load_plugin_readme(
    name: str,
    registry_entry: Optional[dict],
    installed_path: Optional[Path],
) -> str:
    """Load README text from a local install path or remote repository."""
    if installed_path is not None:
        text = read_local_readme(installed_path)
        if text:
            return text
    if registry_entry:
        repo = registry_entry.get("repository", "")
        version = registry_entry.get("version", "latest")
        if repo:
            text = fetch_remote_readme(repo, version)
            if text:
                return text
    if installed_path is not None:
        return "No README found in the plugin directory."
    return "No README found."


def _fmt_category(category) -> str:
    if isinstance(category, list):
        return ", ".join(str(c) for c in category)
    return str(category) if category else ""


_INLINE_MD_RE = re.compile(
    r"(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))"
)


def _insert_inline_md(text: tk.Text, line: str, base_tag: Optional[str]) -> None:
    pos = 0
    for match in _INLINE_MD_RE.finditer(line):
        if match.start() > pos:
            chunk = line[pos:match.start()]
            text.insert(tk.END, chunk, base_tag if base_tag else ())
        if match.group(2):
            tags = (base_tag, "md_bold") if base_tag else ("md_bold",)
            text.insert(tk.END, match.group(2), tags)
        elif match.group(3):
            tags = (base_tag, "md_italic") if base_tag else ("md_italic",)
            text.insert(tk.END, match.group(3), tags)
        elif match.group(4):
            tags = (base_tag, "md_code") if base_tag else ("md_code",)
            text.insert(tk.END, match.group(4), tags)
        elif match.group(5):
            tags = (base_tag, "md_link") if base_tag else ("md_link",)
            text.insert(tk.END, match.group(5), tags)
            text.insert(tk.END, f" ({match.group(6)})")
        pos = match.end()
    if pos < len(line):
        text.insert(tk.END, line[pos:], base_tag if base_tag else ())


def _render_markdown(text: tk.Text, content: str, dpi_scale: float = 1.0) -> None:
    """Apply basic markdown formatting to a Tk Text widget."""
    text.tag_configure("md_h1", font=scaled_font(dpi_scale, "Helvetica", 16, weight="bold"))
    text.tag_configure("md_h2", font=scaled_font(dpi_scale, "Helvetica", 14, weight="bold"))
    text.tag_configure("md_h3", font=scaled_font(dpi_scale, "Helvetica", 12, weight="bold"))
    text.tag_configure("md_bold", font=scaled_font(dpi_scale, "Helvetica", 10, weight="bold"))
    _base10 = scaled_font(dpi_scale, "Helvetica", 10)
    text.tag_configure("md_italic", font=(_base10[0], _base10[1], "italic"))
    text.tag_configure("md_code", font=scaled_font(dpi_scale, "Courier", 10), background="#f0f0f0")
    text.tag_configure("md_codeblock", font=scaled_font(dpi_scale, "Courier", 10), background="#f0f0f0")
    text.tag_configure("md_link", underline=True)

    in_code_block = False
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            text.insert(tk.END, line, "md_codeblock")
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line.rstrip("\n"))
        if heading:
            level = len(heading.group(1))
            _insert_inline_md(text, heading.group(2) + "\n", f"md_h{level}")
            continue

        bullet = re.match(r"^(\s*[-*]|\s*\d+\.)\s+(.*)$", line.rstrip("\n"))
        if bullet:
            _insert_inline_md(text, "  \u2022 " + bullet.group(2) + "\n", None)
            continue

        _insert_inline_md(text, line, None)


class _PluginDetailsWindow:
    """Plugin details dialog with rendered README and install actions."""

    def __init__(
        self,
        app: "CommunityPluginsApp",
        name: str,
        status: str,
        registry_entry: Optional[dict],
        install_path: Optional[Path],
        dpi_scale: float = 1.0,
    ) -> None:
        self.app = app
        self.name = name
        self.status = status
        self.registry_entry = registry_entry
        self.install_path = install_path
        entry = registry_entry or {}
        self._repo_url = get_plugin_repository_url(registry_entry, install_path)
        self._dpi_scale = dpi_scale
        self.window = tk.Toplevel(app.window)
        self.window.title(name)
        self.window.transient(app.window)
        self.window.geometry(f"{scaled(700, dpi_scale)}x{scaled(500, dpi_scale)}")
        self.window.minsize(scaled(400, dpi_scale), scaled(300, dpi_scale))
        self.window.bind("<Escape>", lambda _e: self._on_close())
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = ttk.Frame(self.window, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        meta = ttk.Frame(outer)
        meta.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for label, key in (("Author", "author"), ("Version", "version"), ("Description", "description")):
            row = ttk.Frame(meta)
            row.pack(fill=tk.X, anchor=tk.W, pady=1)
            ttk.Label(row, text=f"{label}:", width=12).pack(side=tk.LEFT)
            value = entry.get(key, "") or "\u2014"
            ttk.Label(row, text=value, wraplength=scaled(620, dpi_scale)).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )

        text_frame = ttk.Frame(outer)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.text = ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED, height=max(8, scaled(12, dpi_scale)))
        self.text.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(outer)
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        actions = ttk.Frame(footer)
        actions.pack(side=tk.LEFT)
        if self._repo_url:
            ttk.Button(
                actions,
                text="Open on GitHub",
                command=lambda: webbrowser.open(self._repo_url),
            ).pack(side=tk.LEFT, padx=(0, 6))
        self.btn_install = ttk.Button(actions, text="Install", command=self._on_install)
        self.btn_install.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_uninstall = ttk.Button(actions, text="Uninstall", command=self._on_uninstall)
        self.btn_uninstall.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_update = ttk.Button(actions, text="Update", command=self._on_update)
        self.btn_update.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(footer, text="Close", command=self._on_close).pack(side=tk.RIGHT)

        app._details_windows.append(self)
        self._sync_action_buttons()
        self._set_body("Loading README\u2026")
        self._load_readme_async()

    def _on_close(self) -> None:
        try:
            self.app._details_windows.remove(self)
        except ValueError:
            pass
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def sync_from_app(self) -> None:
        ctx = self.app._plugin_context_for_name(self.name)
        if ctx is None:
            return
        _name, status, entry, install_path = ctx
        self.status = status
        self.registry_entry = entry
        self.install_path = install_path
        self._repo_url = get_plugin_repository_url(entry, install_path)
        self._sync_action_buttons()

    def _sync_action_buttons(self) -> None:
        busy = self.app._busy
        available = self.status == "Available" and self.registry_entry is not None
        installed = self.status.startswith("Installed")
        has_update = (
            installed
            and self.app._update_statuses.get(self.name) == "update-available"
        )
        self.btn_install.state(["!disabled"] if (available and not busy) else ["disabled"])
        self.btn_uninstall.state(["!disabled"] if (installed and not busy) else ["disabled"])
        self.btn_update.state(["!disabled"] if (has_update and not busy) else ["disabled"])

    def _after_plugin_action(self) -> None:
        self.sync_from_app()
        self._reload_readme()

    def _on_install(self) -> None:
        self.app._install_plugin(
            self.name,
            parent=self.window,
            on_done=self._after_plugin_action,
        )

    def _on_uninstall(self) -> None:
        self.app._uninstall_plugin(
            self.name,
            parent=self.window,
            on_done=self._after_plugin_action,
        )

    def _on_update(self) -> None:
        self.app._update_single(
            self.name,
            parent=self.window,
            on_done=self._after_plugin_action,
        )

    def _set_body(self, content: str, *, rendered: bool = False) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        if rendered:
            _render_markdown(self.text, content, self._dpi_scale)
        else:
            self.text.insert(tk.END, content)
        self.text.configure(state=tk.DISABLED)

    def _load_readme_async(self) -> None:
        name = self.name
        registry_entry = self.registry_entry
        install_path = self.install_path

        def worker() -> str:
            return load_plugin_readme(name, registry_entry, install_path)

        def on_done(content: Optional[str], err: Optional[Exception]) -> None:
            try:
                if not self.window.winfo_exists():
                    return
            except tk.TclError:
                return
            if err is not None:
                self._set_body(f"Failed to load README: {err}")
            else:
                self._set_body(content or "No README found.", rendered=True)

        def run() -> None:
            try:
                result = worker()
                self.app.window.after(0, lambda: on_done(result, None))
            except Exception as exc:  # noqa: BLE001
                self.app.window.after(0, lambda: on_done(None, exc))

        threading.Thread(target=run, daemon=True).start()

    def _reload_readme(self) -> None:
        self._set_body("Loading README\u2026")
        self._load_readme_async()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class CommunityPluginsApp:
    """Standalone window for browsing/installing community plugins."""

    _instance: "Optional[CommunityPluginsApp]" = None

    COLUMNS = ("name", "category", "author", "version", "status", "update_status", "path")

    def __init__(self, parent: Optional[tk.Misc] = None):
        self.parent = parent
        self.plugins_dir = get_plugins_dir()
        self._busy = False
        self._registry: list[dict] = []
        self._update_statuses: dict[str, str] = {}
        self._details_windows: list[_PluginDetailsWindow] = []
        self._owns_root = parent is None

        if parent is None:
            self.window: tk.Misc = tk.Tk()
        else:
            self.window = tk.Toplevel(parent)
            try:
                self.window.transient(parent)
            except tk.TclError:
                pass

        self._dpi_scale = get_dpi_scale(self.window, parent=parent)
        self.window.title("AVLite Community Plugins")
        s = self._dpi_scale
        self.window.geometry(f"{scaled(1100, s)}x{scaled(560, s)}")
        self.window.minsize(scaled(800, s), scaled(420, s))
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

        tree_style = ttk.Style(self.window)
        configure_treeview_style(tree_style, "CP", self._dpi_scale)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=self.COLUMNS,
            show="headings",
            selectmode="browse",
            style="CP.Treeview",
        )
        headings = {
            "name": ("Name", 160),
            "category": ("Category", 100),
            "author": ("Author", 120),
            "version": ("Version", 70),
            "status": ("Status", 120),
            "update_status": ("Update", 100),
            "path": ("Repository / Path", 220),
        }
        s = self._dpi_scale
        for col, (label, width) in headings.items():
            self.tree.heading(col, text=label)
            col_width = scaled(width, s)
            self.tree.column(
                col,
                width=col_width,
                minwidth=col_width,
                anchor=tk.W,
                stretch=(col == "path"),
            )

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._update_buttons())
        self.tree.bind("<Double-Button-1>", lambda _e: self._on_show_details())

        # Toolbar
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(8, 0))
        toolbar_row1 = ttk.Frame(toolbar)
        toolbar_row1.pack(fill=tk.X)
        toolbar_row2 = ttk.Frame(toolbar)
        toolbar_row2.pack(fill=tk.X, pady=(4, 0))

        self.btn_refresh = ttk.Button(toolbar_row1, text="Refresh", command=self._refresh_async)
        self.btn_install = ttk.Button(toolbar_row1, text="Install", command=self._on_install)
        self.btn_uninstall = ttk.Button(toolbar_row1, text="Uninstall", command=self._on_uninstall)
        self.btn_update = ttk.Button(toolbar_row1, text="Update", command=self._on_update)
        self.btn_update_all = ttk.Button(toolbar_row1, text="Update All", command=self._on_update_all)
        self.btn_github = ttk.Button(toolbar_row2, text="Open on GitHub", command=self._on_open_github)
        self.btn_open = ttk.Button(toolbar_row2, text="Open Folder", command=self._open_folder)
        self.btn_close = ttk.Button(toolbar_row2, text="Close", command=self._on_close)

        for b in (self.btn_refresh, self.btn_install, self.btn_uninstall, self.btn_update, self.btn_update_all):
            b.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_github.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))
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
                    _fmt_category(entry.get("category", "")),
                    entry.get("author", ""),
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
                values=(name, "", "", "", status, up_st, str(inst["path"])),
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

    def _selected_plugin_context(self) -> Optional[tuple[str, str, Optional[dict], Optional[Path]]]:
        sel = self._selected_entry()
        if not sel:
            return None
        return self._plugin_context_for_name(sel[0])

    def _plugin_context_for_name(
        self, name: str
    ) -> Optional[tuple[str, str, Optional[dict], Optional[Path]]]:
        try:
            status = self.tree.set(name, "status")
        except tk.TclError:
            return None
        entry = next((e for e in self._registry if e["name"] == name), None)
        install_path = None
        if status.startswith("Installed"):
            installed = {p["name"]: p for p in list_installed(self.plugins_dir)}
            inst = installed.get(name)
            if inst is not None:
                install_path = inst["path"]
        return name, status, entry, install_path

    def _sync_details_windows(self, name: Optional[str] = None) -> None:
        live: list[_PluginDetailsWindow] = []
        for win in self._details_windows:
            try:
                if not win.window.winfo_exists():
                    continue
                live.append(win)
                if name is None or win.name == name:
                    win.sync_from_app()
            except tk.TclError:
                continue
        self._details_windows = live

    def _update_buttons(self) -> None:
        sel = self._selected_entry()
        ctx = self._selected_plugin_context()
        installed = sel is not None and sel[1].startswith("Installed")
        available = sel is not None and sel[1] == "Available"
        busy = self._busy
        has_update = (
            installed and sel is not None
            and self._update_statuses.get(sel[0]) == "update-available"
        )
        has_any_update = any(s == "update-available" for s in self._update_statuses.values())
        has_github = ctx is not None and get_plugin_repository_url(ctx[2], ctx[3]) is not None
        self.btn_github.state(["!disabled"] if (has_github and not busy) else ["disabled"])
        self.btn_install.state(["!disabled"] if (available and not busy) else ["disabled"])
        self.btn_uninstall.state(["!disabled"] if (installed and not busy) else ["disabled"])
        self.btn_refresh.state(["disabled"] if busy else ["!disabled"])
        self.btn_update.state(["!disabled"] if (has_update and not busy) else ["disabled"])
        self.btn_update_all.state(["!disabled"] if (has_any_update and not busy) else ["disabled"])
        self._sync_details_windows()

    def _on_show_details(self) -> None:
        ctx = self._selected_plugin_context()
        if ctx is None:
            return
        name, status, entry, install_path = ctx
        _PluginDetailsWindow(
            self,
            name,
            status,
            entry,
            install_path,
            dpi_scale=self._dpi_scale,
        )

    def _on_open_github(self) -> None:
        ctx = self._selected_plugin_context()
        if ctx is None:
            return
        _name, _status, entry, install_path = ctx
        repo_url = get_plugin_repository_url(entry, install_path)
        if repo_url:
            webbrowser.open(repo_url)

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
        self._sync_details_windows(name)

    def _on_update(self) -> None:
        """Update the currently selected plugin."""
        sel = self._selected_entry()
        if not sel or not sel[1].startswith("Installed"):
            return
        name = sel[0]
        if self._update_statuses.get(name) != "update-available":
            return
        self._update_single(name)

    def _update_single(
        self,
        name: str,
        *,
        parent: Optional[tk.Misc] = None,
        on_done=None,
    ) -> None:
        parent = parent or self.window
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
                messagebox.showerror("Update failed", str(err), parent=parent)
                return
            self._update_statuses.pop(name, None)
            self._set_busy(False, f"Updated {name}.")
            self._populate()
            self._notify_host_changed()
            if on_done:
                on_done()

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

    def _install_plugin(
        self,
        name: str,
        *,
        parent: Optional[tk.Misc] = None,
        on_done=None,
    ) -> None:
        parent = parent or self.window
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
                messagebox.showerror("Install failed", str(err), parent=parent)
                return
            self._handle_requirements(name, path, parent=parent)
            self._set_busy(False, f"Installed {name} at {path}")
            self._populate()
            self._notify_host_changed()
            if on_done:
                on_done()

        self._run_bg(task, done)

    def _on_install(self) -> None:
        sel = self._selected_entry()
        if not sel or sel[1] != "Available":
            return
        self._install_plugin(sel[0])

    def _handle_requirements(
        self, name: str, plugin_path: Path, *, parent: Optional[tk.Misc] = None
    ) -> None:
        """Check the plugin's requirements.txt and prompt to install missing deps."""
        parent = parent or self.window
        req_file = plugin_path / "requirements.txt"
        if not req_file.exists():
            return
        missing, mismatched = check_requirements(req_file)
        if mismatched:
            messagebox.showwarning(
                "Dependency version mismatch",
                f"'{name}' requires:\n  " + "\n  ".join(mismatched)
                + "\n\nThe plugin may not work correctly.",
                parent=parent,
            )
        if missing and messagebox.askyesno(
            "Install missing dependencies?",
            f"'{name}' needs:\n  " + "\n  ".join(missing)
            + "\n\nInstall them into the current Python environment?",
            parent=parent,
        ):
            try:
                pip_install(req_file)
            except subprocess.CalledProcessError as e:
                detail = "\n".join(p for p in (e.stdout, e.stderr) if p) or str(e)
                messagebox.showerror(
                    "pip install failed",
                    detail,
                    parent=parent,
                )

    def _uninstall_plugin(
        self,
        name: str,
        *,
        parent: Optional[tk.Misc] = None,
        on_done=None,
    ) -> None:
        parent = parent or self.window
        if not messagebox.askyesno(
            "Uninstall plugin",
            f"Remove '{name}' from the profile and delete its files?",
            parent=parent,
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
                messagebox.showerror("Uninstall failed", str(err), parent=parent)
                return
            self._set_busy(False, f"Uninstalled {name}")
            self._populate()
            self._notify_host_changed()
            if on_done:
                on_done()

        self._run_bg(task, done)

    def _on_uninstall(self) -> None:
        sel = self._selected_entry()
        if not sel or not sel[1].startswith("Installed"):
            return
        self._uninstall_plugin(sel[0])

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
