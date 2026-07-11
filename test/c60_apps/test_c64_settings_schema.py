"""Tests for Pydantic-backed settings validation."""

from __future__ import annotations

from pathlib import Path

import argparse
import pytest
import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings, ExecutionSettingsSchema
from avlite.c60_apps.c65_setting_utils import load_setting, save_setting
from avlite.c60_apps.c64_settings_schema import (
    SettingsValidationError,
    apply_validated_to_setting,
    dump_from_setting,
    field_description,
    field_tooltip_text,
    validate_profile,
)


def test_validate_profile_coerces_string_number():
    model = validate_profile(ExecutionSettingsSchema, {"c40_control_dt": "0.05"})
    assert model.c40_control_dt == 0.05


def test_validate_profile_rejects_bad_type():
    with pytest.raises(SettingsValidationError) as exc_info:
        validate_profile(
            ExecutionSettingsSchema,
            {"c40_control_dt": "not-a-number"},
            filepath="configs/c40_execution.yaml",
            profile="default",
        )
    assert "c40_control_dt" in str(exc_info.value)


def test_validate_profile_ignores_unknown_keys():
    model = validate_profile(
        ExecutionSettingsSchema,
        {"c40_bridge": "BasicSim", "unknown_key": 123},
    )
    assert model.c40_bridge == "BasicSim"


def test_apply_validated_to_class():
    original = ExecutionSettings.c40_control_dt
    try:
        validated = ExecutionSettingsSchema.model_validate({"c40_control_dt": 0.02})
        apply_validated_to_setting(ExecutionSettings, validated)
        assert ExecutionSettings.c40_control_dt == 0.02
    finally:
        ExecutionSettings.c40_control_dt = original


def test_dump_from_class_excludes_filepath():
    data = dump_from_setting(ExecutionSettings, ExecutionSettingsSchema)
    assert "filepath" not in data
    assert "schema" not in data
    assert "c40_bridge" in data


def test_field_description():
    desc = field_description(ExecutionSettings, "c40_control_dt")
    assert desc is not None
    assert "control" in desc.lower()


def test_field_tooltip_text_description_first():
    field = ExecutionSettingsSchema.model_fields["c40_control_dt"]
    expected = (
        f"{field.description} (float, default={field.default!r}, config_name: c40_control_dt)"
    )
    assert field_tooltip_text(ExecutionSettingsSchema, "c40_control_dt") == expected


def test_load_setting_valid_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "default.yaml").write_text(
        yaml.dump({"c40_execution": {"c40_bridge": "BasicSim", "c40_control_dt": 0.01}})
    )
    original_bridge = ExecutionSettings.c40_bridge
    original_dt = ExecutionSettings.c40_control_dt
    try:
        assert load_setting(ExecutionSettings, profile="default") is True
        assert ExecutionSettings.c40_bridge == "BasicSim"
        assert ExecutionSettings.c40_control_dt == 0.01
    finally:
        ExecutionSettings.c40_bridge = original_bridge
        ExecutionSettings.c40_control_dt = original_dt


def test_load_setting_invalid_type(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "default.yaml").write_text(
        yaml.dump({"c40_execution": {"c40_control_dt": "bad"}})
    )
    original_dt = ExecutionSettings.c40_control_dt
    try:
        assert load_setting(ExecutionSettings, profile="default") is False
        assert ExecutionSettings.c40_control_dt == original_dt
    finally:
        ExecutionSettings.c40_control_dt = original_dt


def test_save_setting_round_trip(monkeypatch, tmp_path):
    from avlite.c60_apps.c65_setting_utils import profile_file_path

    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "testprof.yaml").write_text(yaml.dump({"c10_perception": {"c11_max_agents": 5}}))
    original_bridge = ExecutionSettings.c40_bridge
    ExecutionSettings.c40_bridge = "RoundTripBridge"
    try:
        save_setting(ExecutionSettings, profile="testprof")
        write_path = Path(profile_file_path("testprof", for_write=True))
        assert write_path.is_relative_to(tmp_path.resolve())
        with open(tmp_path / "testprof.yaml") as f:
            saved = yaml.safe_load(f)
        assert "c10_perception" in saved
        assert saved["c40_execution"]["c40_bridge"] == "RoundTripBridge"
        assert "filepath" not in saved["c40_execution"]
        model = validate_profile(ExecutionSettingsSchema, saved["c40_execution"])
        assert model.c40_bridge == "RoundTripBridge"
    finally:
        ExecutionSettings.c40_bridge = original_bridge


def _parse_setting_cli_args(argv: list[str]) -> argparse.Namespace:
    from avlite.plugins.p60_setting_cli.p61_setting_cli import configure_parser

    parser = argparse.ArgumentParser(prog="avlite")
    sub = parser.add_subparsers(dest="command")
    setting_cli = sub.add_parser("setting-cli")
    configure_parser(setting_cli)
    return parser.parse_args(argv)


def test_run_setting_command_bare_setting_shows_help(capsys):
    from avlite.plugins.p60_setting_cli.p61_setting_cli import run_setting_command

    args = _parse_setting_cli_args(["setting-cli"])
    assert run_setting_command(args) == 0
    out = capsys.readouterr().out
    assert "validate" in out
    assert "describe" in out


def test_default_map_settings_field():
    from avlite.plugins.p60_visualizer_tk.p65_ui_lib import DataPicker

    assert DataPicker.default_map_settings_field() == "c40_map"


def test_run_setting_command_help_subcommand(capsys):
    from avlite.plugins.p60_setting_cli.p61_setting_cli import run_setting_command

    args = _parse_setting_cli_args(["setting-cli", "help"])
    assert run_setting_command(args) == 0
    out = capsys.readouterr().out
    assert "validate" in out
    assert "describe" in out


def test_tuning_knob_reaches_controller_without_code_reload():
    """Regression: mutating the settings singleton (as a profile load does) must reach
    a freshly built controller via a plain factory/constructor call, with no module reload."""
    from avlite.c30_control.c34_stanley import StanleyController
    from avlite.c30_control.c39_settings import ControlSettings

    original = ControlSettings.c34_stanley_k
    try:
        ControlSettings.c34_stanley_k = original + 3.0
        controller = StanleyController()
        assert controller.k == original + 3.0
    finally:
        ControlSettings.c34_stanley_k = original


def test_plugin_settings_filepath_from_directory_name():
    from avlite.c60_apps.c68_paths import PluginPaths

    assert PluginPaths.settings_filepath("avlite-bridge-carla") == "configs/plugin_avlite-bridge-carla.yaml"
