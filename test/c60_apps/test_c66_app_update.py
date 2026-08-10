"""Tests for PyPI version check and pip upgrade helpers."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from avlite.c60_apps.c66_app_update import AppUpdater


class _FakeResp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def __enter__(self):
        return io.BytesIO(self._raw)

    def __exit__(self, *args):
        return False


def test_latest_returns_info_version():
    with patch(
        "avlite.c60_apps.c66_app_update.urllib.request.urlopen",
        return_value=_FakeResp({"info": {"version": "0.5.4"}}),
    ):
        assert AppUpdater.latest() == "0.5.4"


def test_latest_missing_version_raises():
    with patch(
        "avlite.c60_apps.c66_app_update.urllib.request.urlopen",
        return_value=_FakeResp({"info": {}}),
    ):
        with pytest.raises(ValueError, match="missing info.version"):
            AppUpdater.latest()


def test_is_newer_uses_packaging_when_available():
    assert AppUpdater.is_newer("0.5.4", "0.5.3") is True
    assert AppUpdater.is_newer("0.5.3", "0.5.3") is False
    assert AppUpdater.is_newer("0.5.2", "0.5.3") is False


def test_is_newer_falls_back_when_packaging_fails(monkeypatch):
    import packaging.version as pv

    class BoomVersion:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("no packaging")

    monkeypatch.setattr(pv, "Version", BoomVersion)

    assert AppUpdater.is_newer("0.6.0", "0.5.9") is True
    assert AppUpdater.is_newer("0.5.9", "0.6.0") is False
    # Non-numeric suffixes strip to digits for the fallback comparator.
    assert AppUpdater.is_newer("1.2.0rc1", "1.1.9") is True


def test_upgrade_invokes_pip_and_raises_trimmed_stderr(monkeypatch):
    long_err = "x" * 600

    def fake_run(cmd, **kwargs):
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
        assert "--upgrade" in cmd and "avlite" in cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=long_err)

    monkeypatch.setattr("avlite.c60_apps.c66_app_update.subprocess.run", fake_run)
    with pytest.raises(RuntimeError) as exc_info:
        AppUpdater.upgrade()
    assert len(str(exc_info.value)) == 500
    assert str(exc_info.value) == long_err[-500:]


def test_upgrade_succeeds_on_zero_exit(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("avlite.c60_apps.c66_app_update.subprocess.run", fake_run)
    AppUpdater.upgrade()
