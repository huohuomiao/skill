#!/usr/bin/env python3
"""Small dependency-free JSON Schema validator for the bundled contracts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from validation_common import strict_json_loads


class SchemaValidationError(ValueError):
    """Raised when an instance violates a supported schema rule."""


def _matches_type(instance: Any, expected: str) -> bool:
    mapping = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    return expected in mapping and mapping[expected](instance)


def _resolve_ref(schema_path: Path, reference: str) -> tuple[dict[str, Any], Path]:
    if reference.startswith("#"):
        raise SchemaValidationError("local fragment references are not supported")
    reference_path = (schema_path.parent / reference).resolve()
    try:
        return strict_json_loads(reference_path.read_text(encoding="utf-8")), reference_path
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(f"cannot resolve schema reference {reference}: {exc}") from exc


def _validate_date_time(value: str, location: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{location}: invalid date-time") from exc
    if parsed.tzinfo is None:
        raise SchemaValidationError(f"{location}: date-time must include timezone")


def validate(instance: Any, schema: dict[str, Any], schema_path: Path, location: str = "$") -> None:
    if "$ref" in schema:
        referenced, referenced_path = _resolve_ref(schema_path, schema["$ref"])
        validate(instance, referenced, referenced_path, location)

    for part in schema.get("allOf", []):
        validate(instance, part, schema_path, location)

    if "if" in schema:
        try:
            validate(instance, schema["if"], schema_path, location)
        except SchemaValidationError:
            pass
        else:
            if "then" in schema:
                validate(instance, schema["then"], schema_path, location)

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_matches_type(instance, expected) for expected in expected_types):
            raise SchemaValidationError(f"{location}: expected type {expected_types}")

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{location}: value is not in enum")

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{location}: string is too short")
        if schema.get("format") == "date-time":
            _validate_date_time(instance, location)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{location}: number is below minimum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(f"{location}: number is not above exclusiveMinimum")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaValidationError(f"{location}: missing required fields {missing}")
        properties = schema.get("properties", {})
        unexpected = sorted(set(instance) - set(properties))
        additional = schema.get("additionalProperties", True)
        if additional is False:
            if unexpected:
                raise SchemaValidationError(f"{location}: unexpected fields {unexpected}")
        elif isinstance(additional, dict):
            for key in unexpected:
                validate(instance[key], additional, schema_path, f"{location}.{key}")
        for key, child_schema in properties.items():
            if key in instance:
                validate(instance[key], child_schema, schema_path, f"{location}.{key}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{location}: array has too few items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in instance]
            if len(encoded) != len(set(encoded)):
                raise SchemaValidationError(f"{location}: array items are not unique")
        if "items" in schema:
            for index, item in enumerate(instance):
                validate(item, schema["items"], schema_path, f"{location}[{index}]")


def validate_file(instance_path: Path, schema_path: Path) -> None:
    try:
        instance = strict_json_loads(instance_path.read_text(encoding="utf-8"))
        schema = strict_json_loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(str(exc)) from exc
    validate(instance, schema, schema_path.resolve())
