"""Check PyPI for a newer avlite and optionally pip-upgrade it.

Tk-free; UI layers (e.g. the visualizer) own dialogs and toasts.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import urllib.request

from avlite import __version__

log = logging.getLogger(__name__)

_PYPI_JSON = "https://pypi.org/pypi/avlite/json"
_TIMEOUT_S = 5


class AppUpdater:
    """PyPI check + pip upgrade; no Tk."""

    @staticmethod
    def latest() -> str:
        log.info("Checking PyPI for avlite updates (current %s)…", __version__)
        with urllib.request.urlopen(_PYPI_JSON, timeout=_TIMEOUT_S) as resp:
            data = json.load(resp)
        version = data.get("info", {}).get("version")
        if not version:
            raise ValueError("PyPI response missing info.version")
        latest = str(version)
        log.info("PyPI latest avlite version: %s", latest)
        return latest

    @staticmethod
    def is_newer(latest: str, current: str | None = None) -> bool:
        cur = current if current is not None else __version__
        try:
            from packaging.version import Version

            return Version(latest) > Version(cur)
        except Exception:
            def _parts(v: str) -> tuple[int, ...]:
                out: list[int] = []
                for p in v.split("."):
                    digits = "".join(c for c in p if c.isdigit())
                    out.append(int(digits) if digits else 0)
                return tuple(out)

            return _parts(latest) > _parts(cur)

    @staticmethod
    def upgrade() -> None:
        log.info("Upgrading avlite via pip (%s → latest)…", __version__)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "avlite"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "pip upgrade failed").strip()
            raise RuntimeError(err[-500:] if len(err) > 500 else err)
        log.info("pip upgrade of avlite finished successfully")
