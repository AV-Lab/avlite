"""Ensure stack core layers do not import the app/GUI layer.

Stack layers (``c10``–``c40``, ``c50_common``) must not import ``c60_apps``
(except allowed app-infra modules) or ``p60_*`` Tk app plugins.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_APP_MODULES = (
    "avlite.c60_apps.c61_app_strategy",
    "avlite.c60_apps.c64_settings_schema",
    "avlite.c60_apps.c68_paths",
)

FORBIDDEN_PREFIXES = (
    "avlite.c60_apps",
    "avlite.plugins.p60_",
)

SCAN_ROOTS = (
    ROOT / "avlite" / "c10_perception",
    ROOT / "avlite" / "c20_planning",
    ROOT / "avlite" / "c30_control",
    ROOT / "avlite" / "c50_common",
    ROOT / "avlite" / "c40_execution",
)


def _imports_from_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_core_layers_do_not_import_app():
    violations: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in scan_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for mod in _imports_from_file(path):
                if mod in ALLOWED_APP_MODULES:
                    continue
                if any(mod.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                    violations.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not violations, "\n".join(violations)
