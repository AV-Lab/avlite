"""Pydantic schemas and helpers for validated YAML settings profiles."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, Type

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.fields import FieldInfo

SETTINGS_META = frozenset({"exclude", "filepath", "schema"})


class SettingsBinder(Protocol):
    def get_value(self, setting: Any, field_name: str) -> Any: ...
    def set_value(self, setting: Any, field_name: str, value: Any) -> None: ...


class PlainBinder:
    """Read/write settings attributes as plain Python values (no Tk)."""

    def get_value(self, setting: Any, field_name: str) -> Any:
        attr_value = _resolve_attr(setting, field_name)
        if callable(attr_value) or field_name.startswith("_"):
            raise ValueError(f"Cannot read {field_name}")
        val = attr_value
        if isinstance(val, (np.floating, np.integer)):
            return val.item()
        return val

    def set_value(self, setting: Any, field_name: str, value: Any) -> None:
        attr_value = _resolve_attr(setting, field_name)
        if callable(attr_value):
            return
        coerced = _coerce_numpy_scalar(value)
        if value is None and isinstance(attr_value, (list, dict)):
            setattr(setting, field_name, type(attr_value)())
        else:
            setattr(setting, field_name, coerced)


class SettingsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # YAML location for this settings group. Core layers override with a fixed path;
    # plugins leave it blank and the plugin loader fills it in from the plugin's
    # directory name (see PluginPaths.settings_filepath).
    filepath: ClassVar[str] = ""


def schema_of(setting: Any) -> Type[BaseModel] | None:
    """Resolve the Pydantic schema class for a settings instance, model class, or legacy class.

    - A ``SettingsSchema`` instance (the new singleton form) returns its own type.
    - A ``BaseModel`` subclass returns itself.
    - A legacy plain class (e.g. the Tk-backed visualization settings) returns its
      ``schema`` attribute.
    """
    if isinstance(setting, type):
        if issubclass(setting, BaseModel):
            return setting
        return getattr(setting, "schema", None)
    if isinstance(setting, BaseModel):
        return type(setting)
    return getattr(type(setting), "schema", None)


def setting_key(setting: Any) -> str:
    """Stable identifier for a settings class or singleton instance.

    Instances report their model class name with a trailing ``Schema`` stripped so
    that, e.g., the ``ExecutionSettings`` singleton keys as ``"ExecutionSettings"``.
    """
    name = getattr(setting, "__name__", None) or type(setting).__name__
    return name[:-6] if name.endswith("Schema") else name


def reset_to_defaults(setting: Any) -> None:
    """Re-apply each schema field's default onto *setting* in place (preserving identity)."""
    schema = schema_of(setting)
    if schema is None:
        return
    for name, info in schema.model_fields.items():
        if name in SETTINGS_META:
            continue
        if not hasattr(setting, name):
            continue
        setattr(setting, name, info.get_default(call_default_factory=True))


class SettingsValidationError(Exception):
    """Raised when a profile dict fails schema validation."""

    def __init__(
        self,
        filepath: str,
        profile: str,
        message: str,
        *,
        field: str | None = None,
    ) -> None:
        self.filepath = filepath
        self.profile = profile
        self.field = field
        super().__init__(message)

    def __str__(self) -> str:
        loc = f"{self.filepath} / profile '{self.profile}'"
        if self.field:
            return f"{loc} / {self.field}: {super().__str__()}"
        return f"{loc}: {super().__str__()}"


def _coerce_numpy_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _format_validation_error(
    exc: ValidationError,
    filepath: str,
    profile: str,
    schema: Type[BaseModel] | None = None,
) -> SettingsValidationError:
    errors = exc.errors()
    if not errors:
        return SettingsValidationError(filepath, profile, str(exc))
    first = errors[0]
    field_path = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", str(exc))
    if schema is not None and field_path:
        desc = field_description(schema, field_path)
        if desc:
            msg = f"{msg} — {desc}"
    return SettingsValidationError(filepath, profile, msg, field=field_path or None)


def validate_profile(
    schema: Type[BaseModel],
    profile_dict: dict[str, Any],
    *,
    filepath: str = "",
    profile: str = "default",
) -> BaseModel:
    try:
        return schema.model_validate(profile_dict)
    except ValidationError as exc:
        raise _format_validation_error(exc, filepath, profile, schema) from exc


def _resolve_attr(setting: Any, field_name: str) -> Any:
    if isinstance(setting, type):
        return getattr(setting, field_name)
    return getattr(setting, field_name)


def apply_validated_to_setting(
    setting: Any,
    validated: BaseModel,
    *,
    binder: SettingsBinder | None = None,
) -> None:
    """Apply validated schema fields onto a settings class or instance."""
    bind = binder or PlainBinder()
    for field_name, value in validated.model_dump().items():
        if field_name in SETTINGS_META:
            continue
        if not hasattr(setting, field_name):
            continue
        attr_value = _resolve_attr(setting, field_name)
        if callable(attr_value):
            continue
        bind.set_value(setting, field_name, value)


def _collect_field_values(
    setting: Any,
    schema: Type[BaseModel],
    binder: SettingsBinder,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field_name in schema.model_fields:
        if field_name in SETTINGS_META:
            continue
        if not hasattr(setting, field_name):
            continue
        attr_value = _resolve_attr(setting, field_name)
        if callable(attr_value) or field_name.startswith("_"):
            continue
        data[field_name] = binder.get_value(setting, field_name)
    return data


def dump_from_setting(
    setting: Any,
    schema: Type[BaseModel],
    *,
    filepath: str = "",
    profile: str = "default",
    binder: SettingsBinder | None = None,
) -> dict[str, Any]:
    """Read current settings, validate round-trip, return YAML-safe dict."""
    bind = binder or PlainBinder()
    raw = _collect_field_values(setting, schema, bind)
    validated = validate_profile(schema, raw, filepath=filepath, profile=profile)
    return validated.model_dump()


def field_description(schema_or_cls: Type[BaseModel] | Any, field_name: str) -> str | None:
    """Return Pydantic Field description for a settings field, if any."""
    schema = schema_of(schema_or_cls)
    if schema is None:
        return None
    info: FieldInfo | None = schema.model_fields.get(field_name)
    if info is None:
        return None
    return info.description


def field_tooltip_text(schema_or_cls: Type[BaseModel] | Any, field_name: str) -> str | None:
    """Build tooltip text: description first, then type/default in brackets."""
    schema = schema_of(schema_or_cls)
    if schema is None:
        return None
    info = schema.model_fields.get(field_name)
    if info is None:
        return None
    desc = info.description
    if not desc:
        return None
    annotation = info.annotation
    type_name = getattr(annotation, "__name__", str(annotation))
    suffix = f"({type_name}"
    if not info.is_required():
        suffix += f", default={info.default!r}"
    suffix += f", config_name: {field_name})"
    return f"{desc} {suffix}"


def describe_schema(
    schema: Type[BaseModel],
    *,
    field_filter: str | None = None,
) -> list[str]:
    """Return human-readable lines describing schema fields."""
    lines: list[str] = []
    for name, info in schema.model_fields.items():
        if field_filter is not None and name != field_filter:
            continue
        annotation = info.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        default = info.default
        default_str = ""
        if default is not None and default is not ... and str(default) != "PydanticUndefined":
            default_str = f", default={default!r}"
        lines.append(f"{name} ({type_name}{default_str})")
        if info.description:
            lines.append(f"  {info.description}")
    return lines
