"""Ensure core layers do not import visualization (c50)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_PREFIXES = (
    "avlite.c50_visualization",
)

SCAN_ROOTS = (
    ROOT / "avlite" / "c60_common",
    ROOT / "avlite" / "c40_execution",
    ROOT / "avlite" / "plugins",
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


def test_core_layers_do_not_import_c50():
    violations: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in scan_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for mod in _imports_from_file(path):
                if any(mod.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                    violations.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not violations, "\n".join(violations)
