"""Smoke tests for headless config CLI (avlite.plugins.p50_headless_mode.p52_config_cli).

Tests verify:
- Bare config command prints help and exits cleanly.
- Profile export/import round-trips through a temporary config directory.
"""

import argparse
import zipfile

import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_common.c69_setting_utils import export_profile, import_profile
from avlite.plugins.p50_headless_mode.p52_config_cli import register_config_parser, run_config_command


def test_config_help_exits_zero(capsys):
    parser = argparse.ArgumentParser(prog="avlite")
    sub = parser.add_subparsers(dest="command")
    register_config_parser(sub)
    args = parser.parse_args(["config"])
    assert run_config_command(args) == 0
    captured = capsys.readouterr()
    assert "validate" in captured.out


def test_profile_export_import_round_trip(isolated_config_dir, tmp_path):
    profile_path = isolated_config_dir / "c40_execution.yaml"
    profile_path.write_text(
        yaml.dump({"smoke": {"c40_bridge": "BasicSim", "c40_control_dt": 0.05}})
    )

    zip_path = tmp_path / "smoke.zip"
    export_profile("smoke", zip_path, settings_classes=[ExecutionSettings])

    (isolated_config_dir / "c40_execution.yaml").unlink()
    profile_name = import_profile(
        zip_path, settings_classes=[ExecutionSettings], overwrite=True
    )
    assert profile_name == "smoke"

    reloaded = yaml.safe_load((isolated_config_dir / "c40_execution.yaml").read_text())
    assert "smoke" in reloaded
    assert reloaded["smoke"]["c40_bridge"] == "BasicSim"

    with zipfile.ZipFile(zip_path) as zf:
        assert any(name.endswith(".yaml") for name in zf.namelist())
