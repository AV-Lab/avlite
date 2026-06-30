"""CLI for validating and describing YAML settings profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from avlite.c60_common.c67_paths import ConfigPaths
from avlite.c40_execution.c43_factory import get_stack_settings_classes
from avlite.c60_common.c69_setting_utils import export_profile, import_profile
from avlite.c60_common.c66_plugins import list_plugins, load_builtin_plugin_settings
from avlite.c60_common.c68_settings_schema import (
    SettingsValidationError,
    describe_schema,
    schema_of,
    validate_profile,
)

_LAYER_REGISTRY: dict[str, Any] | None = None
_config_parser: argparse.ArgumentParser | None = None


def _layers() -> dict[str, Any]:
    global _LAYER_REGISTRY
    if _LAYER_REGISTRY is None:
        from avlite.c10_perception.c19_settings import PerceptionSettings
        from avlite.c20_planning.c29_settings import PlanningSettings
        from avlite.c30_control.c39_settings import ControlSettings
        from avlite.c40_execution.c49_settings import ExecutionSettings

        _LAYER_REGISTRY = {
            "perception": PerceptionSettings,
            "planning": PlanningSettings,
            "control": ControlSettings,
            "execution": ExecutionSettings,
        }
    return _LAYER_REGISTRY


def _stack_settings() -> list[Any]:
    return list(_layers().values())


def _plugin_settings() -> list[Any]:
    classes = []
    for plugin in list_plugins():
        cls = load_builtin_plugin_settings(plugin)
        if cls is not None:
            classes.append(cls)
    return classes


def _profiles_in_file(filepath: Path, profile: str | None) -> list[str]:
    if not filepath.exists():
        return []
    with open(filepath) as f:
        data = yaml.safe_load(f) or {}
    if profile is not None:
        return [profile] if profile in data else []
    return list(data.keys())


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    settings_classes = _stack_settings() + _plugin_settings()

    for settings_cls in settings_classes:
        schema = schema_of(settings_cls)
        if schema is None:
            continue
        filepath = Path(ConfigPaths.effective_path(settings_cls.filepath))
        for prof in _profiles_in_file(filepath, args.profile):
            try:
                with open(filepath) as f:
                    config = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                errors.append(f"{filepath}: YAML syntax error: {exc}")
                continue
            profile_dict = config.get(prof, {})
            if not isinstance(profile_dict, dict):
                errors.append(f"{filepath} / profile '{prof}': expected mapping")
                continue
            try:
                validate_profile(schema, profile_dict, filepath=str(filepath), profile=prof)
            except SettingsValidationError as exc:
                errors.append(str(exc))

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print("All profiles valid.")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    layers = _layers()
    if args.layer:
        key = args.layer.lower()
        if key not in layers:
            print(f"Unknown layer '{args.layer}'. Choose from: {', '.join(sorted(layers))}", file=sys.stderr)
            return 1
        schema = schema_of(layers[key])
        for line in describe_schema(schema, field_filter=args.field):
            print(line)
        return 0

    if args.field:
        print(f"Specify --layer with --field; layers: {', '.join(sorted(layers))}", file=sys.stderr)
        return 1

    for name, settings_cls in sorted(layers.items()):
        schema = schema_of(settings_cls)
        if schema is None:
            continue
        print(f"[{name}]")
        for line in describe_schema(schema):
            print(line)
        print()
    return 0


def cmd_help(_args: argparse.Namespace) -> int:
    if _config_parser is not None:
        _config_parser.print_help()
    return 0


def cmd_export_profile(args: argparse.Namespace) -> int:
    output = args.output or f"{args.profile}.zip"
    try:
        from avlite.c40_execution.c49_settings import ExecutionSettings
        from avlite.c60_common.c69_setting_utils import load_setting

        load_setting(ExecutionSettings, profile=args.profile)
        count = export_profile(
            args.profile,
            output,
            settings_classes=get_stack_settings_classes(),
            community_plugins=ExecutionSettings.c40_community_plugins,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to write zip: {exc}", file=sys.stderr)
        return 1
    print(f"Exported profile '{args.profile}' ({count} file(s)) to {output}")
    return 0


def cmd_import_profile(args: argparse.Namespace) -> int:
    try:
        profile_name = import_profile(
            args.zip_path, settings_classes=get_stack_settings_classes(), overwrite=args.force
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to import profile: {exc}", file=sys.stderr)
        return 1
    print(f"Imported profile '{profile_name}'")
    return 0


def register_config_parser(subparsers: argparse._SubParsersAction) -> None:
    global _config_parser
    config = subparsers.add_parser("config", help="Validate or describe settings profiles")
    _config_parser = config
    config_sub = config.add_subparsers(dest="config_command")

    help_p = config_sub.add_parser("help", help="Show config command usage")
    help_p.set_defaults(config_handler=cmd_help)

    validate = config_sub.add_parser("validate", help="Validate YAML profiles against schemas")
    validate.add_argument("--profile", help="Validate only this profile name")
    validate.set_defaults(config_handler=cmd_validate)

    describe = config_sub.add_parser("describe", help="Print field types, defaults, and descriptions")
    describe.add_argument("--layer", help="Stack layer: perception, planning, control, execution")
    describe.add_argument("--field", help="Single field name to describe (requires --layer)")
    describe.set_defaults(config_handler=cmd_describe)

    export_p = config_sub.add_parser("export-profile", help="Export a profile to a zip file")
    export_p.add_argument("profile", help="Profile name to export")
    export_p.add_argument("-o", "--output", help="Output zip path (default: {profile}.zip)")
    export_p.set_defaults(config_handler=cmd_export_profile)

    import_p = config_sub.add_parser("import-profile", help="Import a profile from a zip file")
    import_p.add_argument("zip_path", help="Path to profile zip file")
    import_p.add_argument("--force", action="store_true", help="Overwrite existing profile keys")
    import_p.set_defaults(config_handler=cmd_import_profile)


def run_config_command(args: argparse.Namespace) -> int:
    handler = getattr(args, "config_handler", None)
    if handler is None:
        if _config_parser is not None:
            _config_parser.print_help()
        return 0
    return handler(args)
