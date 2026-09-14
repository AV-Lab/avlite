"""Tests for PluginEnv distro resolution, registry checks, and launch.sh."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from avlite.c60_apps.c67_plugin_env import PluginEnv
from avlite.c60_apps.c69_settings import AppSettings


@pytest.fixture
def ros_root(tmp_path, monkeypatch):
    root = tmp_path / "opt" / "ros"
    monkeypatch.setattr(PluginEnv, "PREFIX", root)
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    monkeypatch.delenv("AVLITE_ROS_DISTRO", raising=False)
    monkeypatch.delenv("AVLITE_ROS_SOURCED", raising=False)
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "")
    PluginEnv.launch_sh = None
    PluginEnv._proc = None
    return root


def _distro(root, name: str) -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "setup.bash").write_text("# test\n", encoding="utf-8")


def test_latest_installed(ros_root):
    _distro(ros_root, "humble")
    _distro(ros_root, "jazzy")
    env = PluginEnv()
    assert env.active == "jazzy"
    assert env.installed == ("humble", "jazzy")


def test_latest_label_and_empty_pick_newest(ros_root, monkeypatch):
    _distro(ros_root, "humble")
    _distro(ros_root, "jazzy")
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "")
    assert PluginEnv().active == "jazzy"
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "latest")
    assert PluginEnv().active == "jazzy"
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "humble")
    assert PluginEnv().active == "humble"


def test_configured_override(ros_root, monkeypatch):
    _distro(ros_root, "humble")
    _distro(ros_root, "jazzy")
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "humble")
    assert PluginEnv().active == "humble"
    monkeypatch.setattr(AppSettings, "c60_ros_distro", "")
    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")
    assert PluginEnv().active == "humble"


def test_keep_already_sourced(ros_root, monkeypatch):
    _distro(ros_root, "humble")
    _distro(ros_root, "jazzy")
    monkeypatch.setenv("ROS_DISTRO", "humble")
    assert PluginEnv().active == "humble"


def test_check_require_ros_false_ignores_range(ros_root):
    PluginEnv().check({"require_ros": False, "min_ros_version": "jazzy"})
    PluginEnv().check({})


def test_check_missing_ros(ros_root):
    with pytest.raises(PluginEnv.Error, match="Active distro: none"):
        PluginEnv().check({"require_ros": True, "name": "demo", "min_ros_version": "humble"})


def test_check_range(ros_root):
    _distro(ros_root, "humble")
    _distro(ros_root, "jazzy")
    env = PluginEnv()
    env.check({"require_ros": True, "min_ros_version": "humble"})
    env.check({"require_ros": True, "min_ros_version": "humble", "max_ros_version": "jazzy"})
    with pytest.raises(PluginEnv.Error, match="humble–humble"):
        env.check(
            {
                "require_ros": True,
                "display_name": "ROS Plugin",
                "min_ros_version": "humble",
                "max_ros_version": "humble",
            }
        )


def test_apply_is_noop_without_setup(ros_root, monkeypatch):
    import os

    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")
    PluginEnv().apply()
    assert os.environ.get("ROS_DISTRO") != "humble"


def test_apply_prepends_pythonpath_to_sys_path(ros_root, tmp_path, monkeypatch):
    import os
    import sys

    _distro(ros_root, "humble")
    site = tmp_path / "ros-site-packages"
    site.mkdir()
    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        env = f"ROS_DISTRO=humble\0PYTHONPATH={site}\0"
        return SimpleNamespace(stdout=env.encode("utf-8"))

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.run", fake_run)
    try:
        PluginEnv().apply()
        assert os.environ.get("ROS_DISTRO") == "humble"
        assert sys.path[0] == str(site)
        assert calls
        cmd, kwargs = calls[0]
        assert cmd[:2] == ["bash", "-c"]
        assert kwargs.get("timeout") == PluginEnv.APPLY_TIMEOUT_S
    finally:
        os.environ.pop("ROS_DISTRO", None)
        if str(site) in sys.path:
            sys.path.remove(str(site))


def test_apply_sources_when_ros_distro_already_matches(ros_root, tmp_path, monkeypatch):
    import os
    import sys

    _distro(ros_root, "humble")
    site = tmp_path / "already-sourced-site"
    site.mkdir()
    monkeypatch.setenv("ROS_DISTRO", "humble")
    monkeypatch.setenv("PYTHONPATH", str(site))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        env = f"ROS_DISTRO=humble\0PYTHONPATH={site}\0"
        return SimpleNamespace(stdout=env.encode("utf-8"))

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.run", fake_run)
    try:
        PluginEnv().apply()
        assert calls
        cmd, kwargs = calls[0]
        assert cmd[:2] == ["bash", "-c"]
        assert kwargs.get("timeout") == PluginEnv.APPLY_TIMEOUT_S
        assert str(site) in sys.path
        assert os.environ.get("ROS_DISTRO") == "humble"
    finally:
        if str(site) in sys.path:
            sys.path.remove(str(site))


def test_apply_reexecs_when_lib_path_missing(ros_root, tmp_path, monkeypatch):
    import os

    _distro(ros_root, "humble")
    lib = ros_root / "humble" / "lib"
    lib.mkdir()
    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    def fake_run(cmd, **kwargs):
        env = f"ROS_DISTRO=humble\0LD_LIBRARY_PATH={lib}\0"
        return SimpleNamespace(stdout=env.encode("utf-8"))

    calls = []

    def fake_execvpe(executable, args, env):
        calls.append((executable, list(args), dict(env)))

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.run", fake_run)
    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.os.execvpe", fake_execvpe)
    monkeypatch.setattr(PluginEnv, "_running_under_pytest", staticmethod(lambda: False))
    PluginEnv().apply()
    assert calls
    _exe, argv, env = calls[0]
    assert argv[0] == _exe
    assert env.get("AVLITE_ROS_SOURCED") == "humble"
    assert env.get("ROS_DISTRO") == "humble"
    assert os.environ.get("AVLITE_ROS_SOURCED") == "humble"


def test_apply_skips_reexec_under_pytest(ros_root, monkeypatch):
    import os

    _distro(ros_root, "humble")
    lib = ros_root / "humble" / "lib"
    lib.mkdir()
    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_apply_skips_reexec_under_pytest")

    def fake_run(cmd, **kwargs):
        env = f"ROS_DISTRO=humble\0LD_LIBRARY_PATH={lib}\0"
        return SimpleNamespace(stdout=env.encode("utf-8"))

    calls = []

    def fake_execvpe(executable, args, env):
        calls.append((executable, list(args), dict(env)))

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.run", fake_run)
    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.os.execvpe", fake_execvpe)
    PluginEnv().apply()
    assert calls == []
    assert os.environ.get("AVLITE_ROS_SOURCED") == "humble"
    assert os.environ.get("ROS_DISTRO") == "humble"


def test_apply_timeout_raises_error(ros_root, monkeypatch):
    import subprocess

    _distro(ros_root, "humble")
    monkeypatch.setenv("AVLITE_ROS_DISTRO", "humble")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.run", fake_run)
    with pytest.raises(PluginEnv.Error, match="Timed out sourcing ROS 2 humble"):
        PluginEnv().apply()


def test_bind_and_start_launch(ros_root, tmp_path, monkeypatch):
    plugin = tmp_path / "fake-ros"
    plugin.mkdir()
    (plugin / ".avlite-registry.yaml").write_text("require_ros: false\n", encoding="utf-8")
    (plugin / "launch.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"fake-ros": str(plugin)})

    class _World:
        __module__ = "avlite.plugins.fake_ros.bridge"

    env = PluginEnv()
    env.bind(SimpleNamespace())  # module will not match
    world = _World()
    env.bind(world)
    assert env.pending_launch == plugin / "launch.sh"

    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(poll=lambda: None, pid=1)

    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.subprocess.Popen", fake_popen)
    env.start_launch()
    assert calls
    assert calls[0][0] == ["bash", str(plugin / "launch.sh")]
    env.start_launch()
    assert len(calls) == 1
