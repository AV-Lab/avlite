"""Smoke tests for setting CLI (avlite.plugins.p60_setting_cli.p61_setting_cli).

Tests verify:
- Bare setting-cli command prints help and exits cleanly.
"""

import argparse

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
