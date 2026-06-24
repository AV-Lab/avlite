"""Tests for Pydantic-backed settings validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import argparse
import pytest
import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings, ExecutionSettingsSchema
from avlite.c60_common.c69_setting_utils import load_setting, save_setting
from avlite.c60_common.c68_settings_schema import (
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
    tip = field_tooltip_text(ExecutionSettings, "c40_control_dt")
    assert tip is not None
    assert tip.startswith("Control loop period")
    assert tip.endswith("(float, default=0.05)")


def test_load_setting_valid_profile(tmp_path):
    filepath = tmp_path / "exec.yaml"
    filepath.write_text(yaml.dump({"default": {"c40_bridge": "BasicSim", "c40_control_dt": 0.01}}))
    original_bridge = ExecutionSettings.c40_bridge
    original_dt = ExecutionSettings.c40_control_dt
    ExecutionSettings.filepath = str(filepath)
    try:
        assert load_setting(ExecutionSettings, profile="default") is True
        assert ExecutionSettings.c40_bridge == "BasicSim"
        assert ExecutionSettings.c40_control_dt == 0.01
    finally:
        ExecutionSettings.filepath = "configs/c40_execution.yaml"
        ExecutionSettings.c40_bridge = original_bridge
        ExecutionSettings.c40_control_dt = original_dt


def test_load_setting_invalid_type(tmp_path):
    filepath = tmp_path / "exec.yaml"
    filepath.write_text(yaml.dump({"default": {"c40_control_dt": "bad"}}))
    original_dt = ExecutionSettings.c40_control_dt
    ExecutionSettings.filepath = str(filepath)
    try:
        assert load_setting(ExecutionSettings, profile="default") is False
        assert ExecutionSettings.c40_control_dt == original_dt
    finally:
        ExecutionSettings.filepath = "configs/c40_execution.yaml"
        ExecutionSettings.c40_control_dt = original_dt


def test_save_setting_round_trip(tmp_path):
    filepath = tmp_path / "exec.yaml"
    filepath.write_text(yaml.dump({"other": {"c40_bridge": "X"}}))
    original_bridge = ExecutionSettings.c40_bridge
    ExecutionSettings.filepath = str(filepath)
    ExecutionSettings.c40_bridge = "RoundTripBridge"
    try:
        save_setting(ExecutionSettings, profile="testprof")
        with open(filepath) as f:
            saved = yaml.safe_load(f)
        assert "other" in saved
        assert saved["testprof"]["c40_bridge"] == "RoundTripBridge"
        assert "filepath" not in saved["testprof"]
        model = validate_profile(ExecutionSettingsSchema, saved["testprof"])
        assert model.c40_bridge == "RoundTripBridge"
    finally:
        ExecutionSettings.filepath = "configs/c40_execution.yaml"
        ExecutionSettings.c40_bridge = original_bridge


def _parse_config_args(argv: list[str]) -> argparse.Namespace:
    from avlite.plugins.p50_headless_mode.p52_config_cli import register_config_parser

    parser = argparse.ArgumentParser(prog="avlite")
    sub = parser.add_subparsers(dest="command")
    register_config_parser(sub)
    return parser.parse_args(argv)


def test_run_config_command_bare_config_shows_help(capsys):
    from avlite.plugins.p50_headless_mode.p52_config_cli import run_config_command

    args = _parse_config_args(["config"])
    assert run_config_command(args) == 0
    out = capsys.readouterr().out
    assert "validate" in out
    assert "describe" in out


def test_run_config_command_help_subcommand(capsys):
    from avlite.plugins.p50_headless_mode.p52_config_cli import run_config_command

    args = _parse_config_args(["config", "help"])
    assert run_config_command(args) == 0
    out = capsys.readouterr().out
    assert "validate" in out
    assert "describe" in out
