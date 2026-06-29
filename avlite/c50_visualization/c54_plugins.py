"""Community plugins browser/installer app for AVLite.

Browses the AV-Lab community plugin registry, installs/uninstalls plugins
into a user-data directory, and (de)registers them with the active
execution profile.

Standalone:  ``avlite plugins`` (or ``python -m avlite plugins``)
Embedded:    ``CommunityPluginsApp.open(parent)`` from the main app.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Optional

import yaml

from avlite.c50_visualization.c58_ui_lib import (
    BUTTON_TOOLTIPS,
    attach_tooltip,
    apply_ttk_theme,
    configure_treeview_style,
    get_dpi_scale,
    scaled,
    scaled_font,
    setup_dpi,
)
from avlite.c60_common.c67_paths import effective_config_path, get_config_dir, get_plugins_dir

log = logging.getLogger(__name__)

REGISTRY_URL = (
    "https://raw.githubusercontent.com/AV-Lab/avlite-community-plugins/main/plugins.yaml"
)
REGISTRY_REPO_URL = "https://github.com/AV-Lab/avlite-community-plugins"
PRIVATE_REGISTRY_REPO = "AV-Lab/avlite-private-plugins"
PRIVATE_REGISTRY_REPO_URL = f"https://github.com/{PRIVATE_REGISTRY_REPO}"
GITHUB_OAUTH_CLIENT_ID = os.environ.get("AVLITE_GITHUB_OAUTH_CLIENT_ID", "Ov23liIWaroX8HeDQh3k")

_COMMUNITY_DISCLAIMER = (
    "Community plugins are third-party code. AV-Lab does not vet or guarantee their safety. "
    "For research and development only (use at your own risk)."
)
_MEMBERS_DISCLAIMER = (
    "Member plugins are AV-Lab–listed but not safety-certified. "
    "For research and development only (use at your own risk)."
)

_INLINE_MD_RE = re.compile(
    r"(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))"
)


class GitHubApiError(Exception):
    """GitHub REST API failure with optional SAML SSO authorize URL."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        sso_url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.sso_url = sso_url


def _parse_github_sso_url(header: str) -> Optional[str]:
    """Extract SSO authorization URL from ``X-GitHub-SSO`` response header."""
    if not header:
        return None
    match = re.search(r"url=(\S+)", header)
    return match.group(1) if match else None


def _github_token_path() -> Path:
    return get_config_dir() / "github_oauth.json"


def _load_github_token() -> Optional[tuple[str, str]]:
    path = _github_token_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        token = data.get("token", "")
        login = data.get("login", "")
        if token and _github_user(token):
            return token, login or _github_user(token) or ""
    except Exception:
        log.debug("Invalid saved GitHub token", exc_info=True)
    _clear_github_token()
    return None


def _save_github_token(token: str, login: str) -> None:
    path = _github_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"token": token, "login": login}), encoding="utf-8")
    path.chmod(0o600)


def _clear_github_token() -> None:
    try:
        _github_token_path().unlink(missing_ok=True)
    except OSError:
        pass


def _github_form_post(url: str, data: dict[str, str], timeout: float = 30.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "User-Agent": "avlite"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _github_api(url: str, token: str, *, accept: Optional[str] = None, timeout: float = 30.0) -> bytes:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "avlite"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("message") or body
        except json.JSONDecodeError:
            message = body or str(exc)
        sso_url = _parse_github_sso_url(exc.headers.get("X-GitHub-SSO", ""))
        raise GitHubApiError(str(message), status=exc.code, sso_url=sso_url) from exc


def _github_user(token: str) -> Optional[str]:
    try:
        body = json.loads(_github_api("https://api.github.com/user", token).decode())
        login = body.get("login")
        return str(login) if login else None
    except Exception:
        return None


def _start_device_flow() -> dict:
    if not GITHUB_OAUTH_CLIENT_ID:
        raise ValueError(
            "GitHub OAuth is not configured. Set AVLITE_GITHUB_OAUTH_CLIENT_ID "
            "to your AV-Lab OAuth app client id."
        )
    return _github_form_post(
        "https://github.com/login/device/code",
        {"client_id": GITHUB_OAUTH_CLIENT_ID, "scope": "repo"},
    )


def _poll_device_flow(device_code: str, interval: int, expires_in: int) -> str:
    deadline = time.monotonic() + expires_in
    wait = max(1, interval)
    while time.monotonic() < deadline:
        data = _github_form_post(
            "https://github.com/login/oauth/access_token",
            {
                "client_id": GITHUB_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        token = data.get("access_token")
        if token:
            return str(token)
        err = data.get("error")
        if err == "authorization_pending":
            time.sleep(wait)
            continue
        if err == "slow_down":
            wait += 5
            time.sleep(wait)
            continue
        raise RuntimeError(data.get("error_description") or err or "GitHub sign-in failed")
    raise TimeoutError("GitHub sign-in timed out")


def _git_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _git_auth_args(token: Optional[str]) -> list[str]:
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    return ["-c", f"http.extraHeader=Authorization: Basic {basic}"]


def _authenticated_clone_url(repository: str, token: Optional[str]) -> str:
    """Return an HTTPS clone URL with embedded token (not stored in git config)."""
    if not token:
        return repository
    parsed = _parse_github_repo(_normalize_repo_url(repository))
    if parsed is None:
        return repository
    owner, repo = parsed
    return f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"


def _run_git(
    args: list[str],
    *,
    token: Optional[str] = None,
    timeout: float = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = ["git"] + _git_auth_args(token) + args
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_git_subprocess_env(),
    )


def _format_git_error(err: BaseException) -> str:
    if isinstance(err, subprocess.CalledProcessError):
        detail = "\n".join(p for p in (err.stderr, err.stdout) if p).strip()
        return detail or str(err)
    if isinstance(err, subprocess.TimeoutExpired):
        return f"Git command timed out after {err.timeout}s"
    return str(err)


class _DeviceFlowDialog:
    """Modal dialog for GitHub Device Flow sign-in."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_success: Callable[[str, str], None],
        dpi_scale: float = 1.0,
    ) -> None:
        self._on_success = on_success
        self._cancelled = False
        self.window = tk.Toplevel(parent)
        self.window.title("Sign in with GitHub")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.geometry(f"{scaled(460, dpi_scale)}x{scaled(220, dpi_scale)}")
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="1. Open the GitHub authorization page").pack(anchor=tk.W)
        self._uri_var = tk.StringVar(value="Starting…")
        ttk.Label(outer, textvariable=self._uri_var, wraplength=scaled(420, dpi_scale)).pack(
            anchor=tk.W, pady=(2, 8)
        )
        ttk.Label(outer, text="2. Enter this code:").pack(anchor=tk.W)
        code_row = ttk.Frame(outer)
        code_row.pack(fill=tk.X, pady=(2, 8))
        self._code_var = tk.StringVar(value="…")
        ttk.Label(
            code_row,
            textvariable=self._code_var,
            font=scaled_font(dpi_scale, "Courier", 14, weight="bold"),
        ).pack(side=tk.LEFT, anchor=tk.W)
        self._btn_copy = ttk.Button(code_row, text="Copy", command=self._copy_code, state=tk.DISABLED)
        self._btn_copy.pack(side=tk.LEFT, padx=(6, 0))
        attach_tooltip(self._btn_copy, BUTTON_TOOLTIPS["cp_copy_code"])
        self._status_var = tk.StringVar(value="Waiting for authorization…")
        ttk.Label(outer, textvariable=self._status_var, foreground="#666").pack(anchor=tk.W)

        btns = ttk.Frame(outer)
        btns.pack(fill=tk.X, pady=(12, 0))
        self._btn_open = ttk.Button(btns, text="Open in browser", command=self._open_browser, state=tk.DISABLED)
        self._btn_open.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self._btn_open, BUTTON_TOOLTIPS["cp_sign_in_browser"])
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)

        self._verification_uri = ""
        threading.Thread(target=self._run_flow, daemon=True).start()

    def _copy_code(self) -> None:
        code = self._code_var.get().strip()
        if not code or code == "…":
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(code)
        self._status_var.set("Code copied to clipboard.")

    def _open_browser(self) -> None:
        if self._verification_uri:
            webbrowser.open(self._verification_uri)

    def _on_cancel(self) -> None:
        self._cancelled = True
        try:
            self.window.grab_release()
            self.window.destroy()
        except tk.TclError:
            pass

    def _finish(self, token: Optional[str], login: Optional[str], err: Optional[Exception]) -> None:
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        if self._cancelled:
            return
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        if err is not None:
            messagebox.showerror("Sign-in failed", str(err), parent=self.window)
            self.window.destroy()
            return
        if token and login:
            self._on_success(token, login)
        self.window.destroy()

    def _run_flow(self) -> None:
        try:
            flow = _start_device_flow()
            self._verification_uri = flow["verification_uri"]
            user_code = flow["user_code"]
            device_code = flow["device_code"]
            interval = int(flow.get("interval", 5))
            expires_in = int(flow.get("expires_in", 900))

            def show_code() -> None:
                self._uri_var.set(self._verification_uri)
                self._code_var.set(user_code)
                self._btn_copy.state(["!disabled"])
                self._btn_open.state(["!disabled"])

            self.window.after(0, show_code)
            if self._cancelled:
                return
            token = _poll_device_flow(device_code, interval, expires_in)
            login = _github_user(token) or ""
            _save_github_token(token, login)
            self.window.after(0, lambda t=token, l=login: self._finish(t, l, None))
        except Exception as exc:  # noqa: BLE001
            self.window.after(0, lambda err=exc: self._finish(None, None, err))


class _PluginDetailsWindow:
    """Plugin details dialog with rendered README and install actions."""

    def __init__(
        self,
        app: "_PluginRegistryPanel",
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
            btn_github = ttk.Button(
                actions,
                text="Open on GitHub",
                command=lambda: webbrowser.open(self._repo_url),
            )
            btn_github.pack(side=tk.LEFT, padx=(0, 6))
            attach_tooltip(btn_github, BUTTON_TOOLTIPS["cp_github"])
        self.btn_install = ttk.Button(actions, text="Install", command=self._on_install)
        self.btn_install.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self.btn_install, BUTTON_TOOLTIPS["cp_install"])
        self.btn_uninstall = ttk.Button(actions, text="Uninstall", command=self._on_uninstall)
        self.btn_uninstall.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self.btn_uninstall, BUTTON_TOOLTIPS["cp_uninstall"])
        self.btn_update = ttk.Button(actions, text="Update", command=self._on_update)
        self.btn_update.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self.btn_update, BUTTON_TOOLTIPS["cp_update"])
        btn_close = ttk.Button(footer, text="Close", command=self._on_close)
        btn_close.pack(side=tk.RIGHT)
        attach_tooltip(btn_close, BUTTON_TOOLTIPS["cp_close"])

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
            token = self.app._token if self.app._private else None
            return load_plugin_readme(name, registry_entry, install_path, token=token)

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
                self.app.window.after(0, lambda r=result: on_done(r, None))
            except Exception as exc:  # noqa: BLE001
                self.app.window.after(0, lambda err=exc: on_done(None, err))

        threading.Thread(target=run, daemon=True).start()

    def _reload_readme(self) -> None:
        self._set_body("Loading README\u2026")
        self._load_readme_async()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class _PluginRegistryPanel(ttk.Frame):
    """One registry tab: community or private AV-Lab plugins."""

    COLUMNS = ("name", "category", "author", "version", "status", "update_status", "path")

    def __init__(
        self,
        parent: tk.Misc,
        *,
        private: bool,
        root_window: tk.Misc,
        dpi_scale: float,
        on_close: Callable[[], None],
        host: Optional[tk.Misc] = None,
    ) -> None:
        super().__init__(parent)
        self._private = private
        self.root_window = root_window
        self.window = root_window
        self._host = host
        self._dpi_scale = dpi_scale
        self._on_close = on_close
        self.plugins_dir = get_plugins_dir()
        self._busy = False
        self._registry: list[dict] = []
        self._update_statuses: dict[str, str] = {}
        self._details_windows: list[_PluginDetailsWindow] = []
        self._token: Optional[str] = None
        self._github_login = ""
        if private:
            loaded = _load_github_token()
            if loaded:
                self._token, self._github_login = loaded
        self._build_ui()
        if not private or self._token:
            self._refresh_async()
        else:
            self._populate()
            self.status_var.set("Sign in to browse member plugins.")

    # -- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        disclaimer = _MEMBERS_DISCLAIMER if self._private else _COMMUNITY_DISCLAIMER
        ttk.Label(
            outer,
            text=disclaimer,
            wraplength=scaled(1050, self._dpi_scale),
            foreground="#996633",
        ).pack(anchor=tk.W, pady=(0, 6))

        if self._private:
            auth = ttk.Frame(outer)
            auth.pack(fill=tk.X, pady=(0, 6))
            self._auth_label = ttk.Label(auth, foreground="#666")
            self._auth_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._btn_sign_in = ttk.Button(auth, text="Sign in with GitHub", command=self._on_sign_in)
            attach_tooltip(self._btn_sign_in, BUTTON_TOOLTIPS["cp_sign_in"])
            self._btn_sign_out = ttk.Button(auth, text="Sign out", command=self._on_sign_out)
            attach_tooltip(self._btn_sign_out, BUTTON_TOOLTIPS["cp_sign_out"])
            self._sync_auth_bar()

        # Tree
        tree_frame = ttk.Frame(outer)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        tree_style = ttk.Style(self.root_window)
        prefix = "CPP" if self._private else "CP"
        configure_treeview_style(tree_style, prefix, self._dpi_scale)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=self.COLUMNS,
            show="headings",
            selectmode="browse",
            style=f"{prefix}.Treeview",
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

        for b, key in (
            (self.btn_refresh, "cp_refresh"),
            (self.btn_install, "cp_install"),
            (self.btn_uninstall, "cp_uninstall"),
            (self.btn_update, "cp_update"),
            (self.btn_update_all, "cp_update_all"),
        ):
            b.pack(side=tk.LEFT, padx=(0, 6))
            attach_tooltip(b, BUTTON_TOOLTIPS[key])
        self.btn_github.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self.btn_github, BUTTON_TOOLTIPS["cp_github"])
        self.btn_open.pack(side=tk.LEFT, padx=(0, 6))
        attach_tooltip(self.btn_open, BUTTON_TOOLTIPS["cp_open_folder"])
        self.btn_close.pack(side=tk.RIGHT)
        attach_tooltip(self.btn_close, BUTTON_TOOLTIPS["cp_close"])

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status_var, anchor=tk.W).pack(
            fill=tk.X, pady=(6, 0)
        )

        self._update_buttons()

    def _sync_auth_bar(self) -> None:
        if not self._private:
            return
        if self._token:
            self._auth_label.config(text=f"Signed in as @{self._github_login}")
            self._btn_sign_in.pack_forget()
            self._btn_sign_out.pack(side=tk.RIGHT)
        else:
            self._auth_label.config(text="Sign in with GitHub to browse member-only plugins.")
            self._btn_sign_out.pack_forget()
            self._btn_sign_in.pack(side=tk.RIGHT)

    def _on_sign_in(self) -> None:
        _DeviceFlowDialog(
            self.root_window,
            on_success=self._on_signed_in,
            dpi_scale=self._dpi_scale,
        )

    def _on_signed_in(self, token: str, login: str) -> None:
        self._token = token
        self._github_login = login
        self._sync_auth_bar()
        self._refresh_async()

    def _on_sign_out(self) -> None:
        _clear_github_token()
        self._token = None
        self._github_login = ""
        self._registry = []
        self._update_statuses.clear()
        self._sync_auth_bar()
        self._populate()
        self.status_var.set("Signed out.")

    def _signed_in(self) -> bool:
        return not self._private or bool(self._token)

    def _auth_token(self) -> Optional[str]:
        return self._token if self._private else None

    # -- Population ------------------------------------------------------
    def _populate(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if self._private and not self._signed_in():
            self._update_buttons()
            return
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
        signed_in = self._signed_in()
        has_update = (
            installed and sel is not None
            and self._update_statuses.get(sel[0]) == "update-available"
        )
        has_any_update = any(s == "update-available" for s in self._update_statuses.values())
        has_github = ctx is not None and get_plugin_repository_url(ctx[2], ctx[3]) is not None
        enabled = signed_in and not busy
        self.btn_github.state(["!disabled"] if (has_github and enabled) else ["disabled"])
        self.btn_install.state(["!disabled"] if (available and enabled) else ["disabled"])
        self.btn_uninstall.state(["!disabled"] if (installed and enabled) else ["disabled"])
        self.btn_refresh.state(["disabled"] if busy or not signed_in else ["!disabled"])
        self.btn_update.state(["!disabled"] if (has_update and enabled) else ["disabled"])
        self.btn_update_all.state(["!disabled"] if (has_any_update and enabled) else ["disabled"])
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
                self.window.after(0, lambda r=result: on_done(r, None))
            except Exception as exc:  # noqa: BLE001
                log.exception("Background task failed")
                self.window.after(0, lambda err=exc: on_done(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_async(self) -> None:
        if not self._signed_in():
            self._registry = []
            self._populate()
            self.status_var.set("Sign in to browse member plugins.")
            return
        self._set_busy(True, "Fetching registry…")
        token = self._auth_token()
        private = self._private

        def task():
            return fetch_registry(private=private, token=token)

        def done(result, err):
            if err is not None:
                self._registry = []
                self._populate()
                msg = str(err)
                if isinstance(err, GitHubApiError):
                    msg = str(err)
                    if err.sso_url:
                        self._set_busy(False, f"Failed to load registry: {msg}")
                        if messagebox.askyesno(
                            "Authorize for AV-Lab",
                            f"{msg}\n\nOpen the GitHub page to authorize this token for AV-Lab, "
                            "then click Refresh.",
                            parent=self.root_window,
                        ):
                            webbrowser.open(err.sso_url)
                        return
                    if err.status == 401:
                        _clear_github_token()
                        self._token = None
                        self._github_login = ""
                        self._sync_auth_bar()
                        msg = "GitHub session expired. Sign in again."
                    elif err.status == 403:
                        msg = (
                            f"{msg}\n\nIf this persists, ask an AV-Lab org admin to approve "
                            "the AVLite OAuth app under Organization Settings → Third-party access."
                        )
                elif isinstance(err, urllib.error.HTTPError) and err.code == 403:
                    msg = "Your GitHub account does not have access to member plugins."
                elif self._private and self._token and "401" in msg:
                    _clear_github_token()
                    self._token = None
                    self._github_login = ""
                    self._sync_auth_bar()
                    msg = "GitHub session expired. Sign in again."
                self._set_busy(False, f"Failed to load registry: {msg}")
                return
            self._registry = result or []
            self._update_statuses.clear()
            self._populate()
            label = "member" if private else "community"
            self._set_busy(False, f"Loaded {len(self._registry)} {label} plugin(s).")

        self._run_bg(task, done)

    def _check_updates_async(self) -> None:
        """Start background update checks for installed plugins not yet checked."""
        installed = {p["name"]: p for p in list_installed(self.plugins_dir)}
        registry_by_name = {e["name"]: e for e in self._registry}
        names_to_check = [
            n for n in registry_by_name
            if n in installed and n not in self._update_statuses
        ]
        if not names_to_check:
            return

        def run_all():
            for name in names_to_check:
                plugin_path = installed[name]["path"]
                registry_entry = registry_by_name.get(name)
                try:
                    result = check_plugin_update(
                        plugin_path, registry_entry, token=self._auth_token()
                    )
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
            update_plugin(plugin_path, version, token=self._auth_token())

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
                    update_plugin(path, version, token=self._auth_token())
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
        host = self._host
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
            path = install_plugin(entry, self.plugins_dir, token=self._auth_token())
            register_in_profile(name, path, profile=profile)
            return path

        def done(path, err):
            if err is not None:
                msg = _format_git_error(err)
                self._set_busy(False, f"Install failed: {msg}")
                messagebox.showerror("Install failed", msg, parent=parent)
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
        host = self._host
        if host is None:
            return
        try:
            cfg_view = getattr(host, "config_shortcut_view", None)
            if cfg_view is not None and hasattr(cfg_view, "update_setting_window"):
                cfg_view.update_setting_window()
        except Exception:
            log.debug("Host refresh hook failed", exc_info=True)


class CommunityPluginsApp:
    """Standalone window for browsing/installing community and private plugins."""

    _instance: "Optional[CommunityPluginsApp]" = None

    def __init__(self, parent: Optional[tk.Misc] = None) -> None:
        self.parent = parent
        self._owns_root = parent is None

        if parent is None:
            setup_dpi()
            self.window: tk.Misc = tk.Tk()
            apply_ttk_theme(self.window, dark=True)
        else:
            self.window = tk.Toplevel(parent)
            try:
                self.window.transient(parent)
            except tk.TclError:
                pass

        self._dpi_scale = get_dpi_scale(self.window, parent=parent)
        self.window.title("AVLite Plugins")
        s = self._dpi_scale
        self.window.geometry(f"{scaled(1100, s)}x{scaled(560, s)}")
        self.window.minsize(scaled(800, s), scaled(420, s))
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.bind("<Escape>", lambda _e: self._on_close())

        try:
            bg = ttk.Style(self.window).lookup("TFrame", "background")
            if bg:
                self.window.configure(background=bg)
        except tk.TclError:
            pass

        outer = ttk.Frame(self.window, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text=f"Plugins directory: {get_plugins_dir()}",
            foreground="#666",
        ).pack(anchor=tk.W, pady=(0, 6))

        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True)

        community_frame = ttk.Frame(nb)
        private_frame = ttk.Frame(nb)
        nb.add(community_frame, text="Community")
        nb.add(private_frame, text="Members")

        panel_kw = dict(
            root_window=self.window,
            dpi_scale=self._dpi_scale,
            on_close=self._on_close,
            host=parent,
        )
        self._community_panel = _PluginRegistryPanel(
            community_frame, private=False, **panel_kw
        )
        self._community_panel.pack(fill=tk.BOTH, expand=True)
        self._private_panel = _PluginRegistryPanel(
            private_frame, private=True, **panel_kw
        )
        self._private_panel.pack(fill=tk.BOTH, expand=True)

    def _on_close(self) -> None:
        CommunityPluginsApp._instance = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass

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


# ---------------------------------------------------------------------------
# Module functions
# ---------------------------------------------------------------------------
def fetch_registry(
    *,
    private: bool = False,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> list[dict]:
    """Fetch and parse a plugins registry (public raw URL or private GitHub API)."""
    if private:
        if not token:
            raise ValueError("GitHub token required for private registry")
        url = (
            f"https://api.github.com/repos/{PRIVATE_REGISTRY_REPO}/contents/plugins.yaml"
            "?ref=main"
        )
        raw = _github_api(url, token, accept="application/vnd.github.raw", timeout=timeout)
        data = yaml.safe_load(raw) or {}
    else:
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


def install_plugin(entry: dict, plugins_dir: Path, *, token: Optional[str] = None) -> Path:
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
    clone_url = _authenticated_clone_url(repo, token)
    if version == "latest":
        _run_git(["clone", "--depth", "1", clone_url, str(target)], timeout=120)
    else:
        _run_git(["clone", clone_url, str(target)], timeout=120)

    clean_url = _normalize_repo_url(repo)
    if _parse_github_repo(clean_url):
        subprocess.run(
            ["git", "-C", str(target), "remote", "set-url", "origin", clean_url],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_git_subprocess_env(),
        )

    if version and version != "latest":
        log.info("Checking out version %s", version)
        _run_git(["-C", str(target), "checkout", version], timeout=60)
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


def get_remote_sha(repo_url: str, ref: str = "HEAD", *, token: Optional[str] = None) -> Optional[str]:
    """Return the commit SHA for *ref* on the remote, or ``None`` on failure.

    Uses ``ref^{}`` to dereference annotated tags to the underlying commit SHA.
    The last matching line from ``git ls-remote`` is used.
    """
    try:
        result = _run_git(
            ["ls-remote", repo_url, ref, f"{ref}^{{}}"],
            token=token,
            timeout=20,
            check=False,
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


def check_plugin_update(
    plugin_path: Path,
    registry_entry: Optional[dict],
    *,
    token: Optional[str] = None,
) -> str:
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
    remote_sha = get_remote_sha(repo, ref, token=token)
    if remote_sha is None:
        return "unknown"
    return "up-to-date" if local_sha == remote_sha else "update-available"


def update_plugin(
    plugin_path: Path,
    version: str = "latest",
    *,
    token: Optional[str] = None,
) -> None:
    """Pull the latest commit, or fetch and check out a specific tag."""
    if version == "latest":
        _run_git(["-C", str(plugin_path), "pull"], token=token, timeout=120)
    else:
        _run_git(["-C", str(plugin_path), "fetch", "origin"], token=token, timeout=120)
        _run_git(["-C", str(plugin_path), "checkout", version], timeout=60)


def _current_profile() -> str:
    """Best-effort: read the active profile from the visualization settings file."""
    try:
        from avlite.c50_visualization.c59_settings import VisualizationSettings

        path = Path(effective_config_path(VisualizationSettings.filepath, for_write=False))
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
    from avlite.c60_common.c67_paths import normalize_community_plugin_stored
    from avlite.c60_common.c69_setting_utils import load_setting, save_setting

    profile = profile or _current_profile()
    load_setting(ExecutionSettings, profile=profile)
    ExecutionSettings.c40_community_plugins[name] = normalize_community_plugin_stored(
        name, str(path)
    )
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


def fetch_remote_readme(
    repository: str,
    version: str = "latest",
    *,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> str:
    """Fetch README.md from a GitHub repository."""
    parsed = _parse_github_repo(repository)
    if parsed is None:
        return ""
    owner, repo = parsed
    refs = [version] if version and version != "latest" else ["main", "master", "HEAD"]
    readme_names = ("README.md", "readme.md")
    if token:
        for ref in refs:
            for readme_name in readme_names:
                url = (
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{readme_name}"
                    f"?ref={ref}"
                )
                try:
                    raw = _github_api(url, token, accept="application/vnd.github.raw", timeout=timeout)
                    return raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
        return ""
    for ref in refs:
        for readme_name in readme_names:
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
    *,
    token: Optional[str] = None,
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
            text = fetch_remote_readme(repo, version, token=token)
            if text:
                return text
    if installed_path is not None:
        return "No README found in the plugin directory."
    return "No README found."


def _fmt_category(category) -> str:
    if isinstance(category, list):
        return ", ".join(str(c) for c in category)
    return str(category) if category else ""



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


def main() -> None:
    """Entry point for ``avlite plugins``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    CommunityPluginsApp.open(parent=None)


if __name__ == "__main__":
    main()
