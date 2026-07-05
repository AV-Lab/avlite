"""Smoke tests for setting CLI (avlite.plugins.p60_setting_cli.p61_setting_cli).

Tests verify:
- Bare setting-cli command prints help and exits cleanly.
- Profile export/import round-trips through a temporary config directory.
"""

import argparse

import yaml

from avlite.c60_apps.c65_setting_utils import export_profile, import_profile
from avlite.plugins.p60_setting_cli.p61_setting_cli import configure_parser, run_setting_command


def test_setting_cli_help_exits_zero(capsys):
    parser = argparse.ArgumentParser(prog="avlite")
    sub = parser.add_subparsers(dest="command")
    setting_cli = sub.add_parser("setting-cli")
    configure_parser(setting_cli)
    args = parser.parse_args(["setting-cli"])
    assert run_setting_command(args) == 0
    captured = capsys.readouterr()
    assert "validate" in captured.out


def test_profile_export_import_round_trip(isolated_config_dir, tmp_path):
    (isolated_config_dir / "smoke.yaml").write_text(
        yaml.dump({"c40_execution": {"c40_bridge": "BasicSim", "c40_control_dt": 0.05}})
    )

    out_path = tmp_path / "smoke.yaml"
    export_profile("smoke", out_path)

    (isolated_config_dir / "smoke.yaml").unlink()
    profile_name = import_profile(out_path, overwrite=True)
    assert profile_name == "smoke"

    reloaded = yaml.safe_load((isolated_config_dir / "smoke.yaml").read_text())
    assert "c40_execution" in reloaded
    assert reloaded["c40_execution"]["c40_bridge"] == "BasicSim"
