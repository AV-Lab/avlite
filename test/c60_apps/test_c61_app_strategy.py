"""Tests for AppStrategy registration in c61_app_strategy."""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps
from avlite.c60_apps.c69_settings import AppSettings


@pytest.fixture(autouse=True)
def _skip_ros_apply(monkeypatch):
    monkeypatch.setattr("avlite.c60_apps.c67_plugin_env.PluginEnv.apply", lambda self: None)


@pytest.fixture(autouse=True)
def _restore_app_registry():
    before = dict(AppStrategy.registry)
    before_mods = set(sys.modules)
    yield
    for key, cls in before.items():
        AppStrategy.registry[key] = cls
    for key in list(AppStrategy.registry):
        if key in before:
            continue
        mod = getattr(AppStrategy.registry[key], "__module__", "")
        if not str(mod).startswith("avlite.plugins.p60"):
            del AppStrategy.registry[key]
    for name in list(sys.modules):
        if name in before_mods:
            continue
        if name.startswith("avlite.plugins.") and not name.startswith("avlite.plugins.p60"):
            del sys.modules[name]


def _write_plugin(plugins: Path, name: str, body: str, *, subdir: str | None = None) -> None:
    repo = plugins / name
    package = repo / subdir if subdir else repo
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    if subdir:
        (repo / ".avlite-registry.yaml").write_text(
            f"plugin_subdir: {subdir}\n", encoding="utf-8"
        )
        (repo / "__init__.py").write_text('"""not the importable package"""\n', encoding="utf-8")


def _app_source(
    class_name: str,
    cli: str,
    help_text: str,
    lines: list[str] | None = None,
    *,
    imports: str = "",
) -> str:
    run_body = "\n".join(f"        {line}" for line in (lines or ["return 0"]))
    header = f"{imports}\n" if imports else ""
    return f"""\
{header}from avlite.c60_apps.c61_app_strategy import AppStrategy

class {class_name}(AppStrategy):
    cli_name = {cli!r}
    help = {help_text!r}

    def run(self, args, unknown):
{run_body}
"""


def test_bootstrap_registers_three_tk_apps(monkeypatch):
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {})
    before = set(AppStrategy.registry)
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"
    added = set(AppStrategy.registry) - before
    assert added <= {None, "setting", "plugins", "headless", "setting-cli"}


def test_load_plugins_false_skips_community_apps(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "lab-tool", _app_source("LabToolApp", "labtool", "Lab tool"))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(AppSettings, "c62_load_plugins", False)
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"lab-tool": "lab-tool"})

    bootstrap_apps()

    assert "labtool" not in AppStrategy.registry


def test_bootstrap_loads_monorepo_plugin_subdir(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    _write_plugin(
        plugins,
        "mono-app",
        _app_source("NestedApp", "nesttool", "nested"),
        subdir="pkg",
    )
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"mono-app": "mono-app"})

    bootstrap_apps()

    cls = AppStrategy.registry["nesttool"]
    assert cls.__name__ == "NestedApp"
    assert cls.__module__ == "avlite.plugins.mono_app"
    import avlite.plugins.mono_app as package

    assert package.__file__.endswith("pkg/__init__.py")


def test_community_loader_exception_warns_and_continues(tmp_path, monkeypatch, caplog):
    plugins = tmp_path / "plugins"
    boom = plugins / "boom-app"
    boom.mkdir(parents=True)
    (boom / "__init__.py").write_text("", encoding="utf-8")
    _write_plugin(plugins, "lab-tool", _app_source("LabToolApp", "labtool", "Lab tool"))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(
        AppSettings,
        "c62_community_plugins",
        {"boom-app": "boom-app", "missing-app": "missing-app", "lab-tool": "lab-tool"},
    )
    from avlite.c60_apps import c63_plugins

    real = c63_plugins.import_plugin_modules

    def wrapped(*args, **kwargs):
        if kwargs.get("pkg_name") == "boom-app":
            raise RuntimeError("nope")
        return real(*args, **kwargs)

    monkeypatch.setattr(c63_plugins, "import_plugin_modules", wrapped)
    with caplog.at_level(logging.WARNING):
        bootstrap_apps()

    assert AppStrategy.registry["labtool"].__name__ == "LabToolApp"
    messages = " ".join(r.message for r in caplog.records)
    assert "boom-app" in messages and "nope" in messages
    assert "missing-app" in messages and "not installed" in messages


def test_community_command_does_not_replace_builtin(tmp_path, monkeypatch, caplog):
    plugins = tmp_path / "plugins"
    _write_plugin(
        plugins,
        "shadow-app",
        _app_source("ShadowPlugins", "plugins", "HIJACK")
        + _app_source("ShadowExtra", "shadowextra", "Extra community command"),
    )
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(AppSettings, "c62_community_plugins", {"shadow-app": "shadow-app"})

    with caplog.at_level(logging.WARNING):
        bootstrap_apps()

    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"
    assert AppStrategy.registry["shadowextra"].__name__ == "ShadowExtra"
    assert any("skipped command 'plugins'" in r.message for r in caplog.records)


def test_main_lists_and_runs_community_command(tmp_path, monkeypatch, capsys, caplog):
    plugins = tmp_path / "plugins"
    marker = tmp_path / "ran"
    _write_plugin(
        plugins,
        "broken-app",
        "raise RuntimeError('broken plugin')\n",
    )
    _write_plugin(
        plugins,
        "lab-tool",
        _app_source(
            "LabToolApp",
            "labtool",
            "Lab tool",
            lines=[f"Path({str(marker)!r}).write_text('ok')", "return 0"],
            imports="from pathlib import Path",
        ),
    )
    config = tmp_path / "config"
    config.mkdir()
    (config / "default.yaml").write_text(
        yaml.dump(
            {
                "c69_apps": {
                    "c62_community_plugins": {
                        "broken-app": "broken-app",
                        "lab-tool": "lab-tool",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    saved = {
        name: getattr(AppSettings, name)
        for name in (
            "c62_load_plugins",
            "c62_default_plugins",
            "c62_community_plugins",
            "c60_selected_profile",
            "c60_profile_note",
            "c60_ros_distro",
        )
    }
    saved["c62_default_plugins"] = list(saved["c62_default_plugins"])
    saved["c62_community_plugins"] = dict(saved["c62_community_plugins"])

    from avlite.__main__ import main

    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SystemExit) as exc:
                main(["--help"])
        assert exc.value.code == 0
        help_out = capsys.readouterr().out
        assert "labtool" in help_out
        assert "Lab tool" in help_out
        assert main(["labtool"]) is None
        assert marker.read_text(encoding="utf-8") == "ok"
        assert any("broken" in r.message.lower() for r in caplog.records)
    finally:
        for name, value in saved.items():
            setattr(AppSettings, name, value)
