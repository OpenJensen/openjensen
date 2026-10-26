"""Small provider protocol, independent from core and actuator packages."""

from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_REQUEST_BYTES = 12 * 1024 * 1024


class ProviderError(Exception):
    """A sanitized error suitable for a caller without exposing HTTP credentials."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fail(message: str) -> None:
    raise ProviderError("invalid_request", message)


def parse_json(raw: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("Duplicate JSON properties are not accepted.")
            result[key] = value
        return result

    def constant(_value):
        fail("Non-finite JSON values are not accepted.")

    try:
        return json.loads(raw, parse_constant=constant, object_pairs_hook=pairs)
    except (ValueError, TypeError, UnicodeDecodeError):
        fail("Request is not valid JSON.")


def number(value: Any, lower: float, upper: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) <= upper + abs(lower)
        and math.isfinite(value)
        and lower <= value <= upper
    )


def validate_arguments(value: Any, joints: list[str]) -> dict:
    if not isinstance(value, dict) or set(value) != {"joint", "delta_rad"}:
        fail("A correction requires exactly joint and delta_rad.")
    if value["joint"] not in joints or not isinstance(value["joint"], str):
        fail("Correction joint is not in the admitted embodiment.")
    if not number(value["delta_rad"], -0.15, 0.15):
        fail("Correction delta must be finite and within 0.15 radians.")
    return {"joint": value["joint"], "delta_rad": float(value["delta_rad"])}


def validate_request(value: Any) -> dict:
    if not isinstance(value, dict):
        fail("Request must be an object.")
    allowed = {
        "operation",
        "provider",
        "model",
        "task",
        "observation",
        "joint_names",
        "max_output_tokens",
        "timeout_seconds",
        "reasoning_effort",
        "continuation",
        "receipt",
        "max_total_tokens",
    }
    if set(value) - allowed:
        fail("Request contains an unsupported property.")
    result = {
        "operation": "plan",
        "model": "gpt-6-astra",
        "max_output_tokens": 1024,
        "timeout_seconds": 30,
        "reasoning_effort": "low",
        "max_total_tokens": 100000,
        "task": "",
        "observation": {},
        **value,
    }
    if result.get("provider") not in {"mock", "openai_responses"}:
        fail("Unsupported provider transport.")
    if result["operation"] not in {"plan", "continue", "check"}:
        fail("Unsupported provider operation.")
    if not isinstance(result["model"], str) or not re.fullmatch(
        r"[a-zA-Z0-9_.:-]{1,128}", result["model"]
    ):
        fail("Invalid model identifier.")
    if not isinstance(result["task"], str) or len(result["task"]) > 16000:
        fail("Task must be at most 16000 characters.")
    if not isinstance(result["observation"], dict):
        fail("Observation must be an object supplied by the host.")
    joints = result.get("joint_names", [])
    if (
        not isinstance(joints, list)
        or not 1 <= len(joints) <= 64
        or any(
            not isinstance(j, str) or not re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,128}", j)
            for j in joints
        )
        or len(set(joints)) != len(joints)
    ):
        if result["operation"] != "check":
            fail("An explicit unique joint manifest is required.")
        result["joint_names"] = ["capability_check_joint"]
    if not number(result["timeout_seconds"], 0.01, 120):
        fail("Timeout must be greater than zero and at most 120 seconds.")
    for field, lower, upper in (
        ("max_output_tokens", 64, 32768),
        ("max_total_tokens", 64, 1000000),
    ):
        if (
            not isinstance(result[field], int)
            or isinstance(result[field], bool)
            or not lower <= result[field] <= upper
        ):
            fail("Invalid token budget.")
    if result["max_output_tokens"] > result["max_total_tokens"]:
        fail("Output reservation exceeds the remaining token budget.")
    if result["reasoning_effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        fail("Unsupported reasoning effort.")
    if result["operation"] == "continue":
        if not isinstance(result.get("continuation"), dict) or not isinstance(
            result.get("receipt"), dict
        ):
            fail("Continuation requires the original context and an execution receipt.")
    return result


def tool_schema(joints: list[str]) -> dict:
    return {
        "type": "function",
        "name": "propose_joint_correction",
        "strict": True,
        "description": "Propose one bounded named simulated joint correction for local admission.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "joint": {"type": "string", "enum": joints},
                "delta_rad": {"type": "number", "minimum": -0.15, "maximum": 0.15},
            },
            "required": ["joint", "delta_rad"],
        },
    }
