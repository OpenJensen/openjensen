"""Bounded scoring records shared by HTTP and durable jobs."""

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL = "muose/Muose-50M-Decision"
REVISION = "5afb8eeff127621fea2d66fc63f56798ada12eda"
MODEL_HASH = "3d81f0712ea1e9495a5996258dbc9a41e2fc2e1912dd87464683b20950e5c76f"
LICENSE = "CC-BY-NC-SA-4.0"
VERSIONS = {"torch": "2.11.0", "safetensors": "0.8.0", "tokenizers": "0.23.2"}
TEMPLATE = "STATE:\n{state}\n\nINSTRUCTIONS:\n{instructions}\n\nCRITERION:\n{criterion}"
# These strings are part of the existing decision worker's exact wire contract.
CAVEATS = [
    "Relative softmax weights are not calibrated correctness probabilities.",
    "Experimental Firebird prompt; not a reproduction of the upstream benchmark.",
    "Banking-tuned text scorer; out-of-domain and robotics quality are unverified.",
    "Advisory only: this output cannot authorize a job, tool, payment, or robot action.",
]
MAX_BYTES = 65536
DEADLINE = 40
CLEANUP_SECONDS = 6


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Criterion(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    text: str = Field(min_length=1, max_length=2000)


class ScoreRequest(Strict):
    schema_version: Literal[1]
    state: str = Field(min_length=1, max_length=8000)
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: list[Criterion] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def text_and_uniqueness(self):
        texts = [self.state, self.instructions, *(item.text for item in self.criteria)]
        for text in texts:
            if not text.strip() or any(ord(c) < 32 and c not in "\n\t" for c in text):
                raise ValueError("Text must contain printable characters")
            text.encode("utf-8")
        if len({item.id for item in self.criteria}) != len(self.criteria) or len(
            {item.text.strip() for item in self.criteria}
        ) != len(self.criteria):
            raise ValueError("Criteria must be unique")
        if len(canonical(self.model_dump())) > MAX_BYTES:
            raise ValueError("Request exceeds limit")
        return self


class Score(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    logit: float
    relative_weight: float = Field(ge=0, le=1)
    tokens: int = Field(ge=1, le=512)


class Timing(Strict):
    load: float = Field(ge=0)
    score: float = Field(ge=0)


class ScoreResult(Strict):
    schema_version: Literal[1]
    model: Literal["muose/Muose-50M-Decision"]
    revision: Literal["5afb8eeff127621fea2d66fc63f56798ada12eda"]
    model_sha256: str
    license: Literal["CC-BY-NC-SA-4.0"]
    prompt_template: Literal["firebird-experimental-sections-v1"]
    prompt_template_sha256: str
    request_sha256: str
    device: Literal["cpu"]
    threads: Literal[2]
    runtime_versions: dict[str, str]
    advisory_only: Literal[True]
    calibrated: Literal[False]
    selected_id: str
    scores: list[Score] = Field(min_length=2, max_length=8)
    timing_ms: Timing
    caveats: list[str]


class DecisionStatus(Strict):
    configured: bool
    available: bool
    busy: bool
    model: str = MODEL
    revision: str = REVISION
    license: str = LICENSE
    advisory_only: Literal[True] = True
    message: str


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    def finite(raw):
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError("Nonfinite number")
        return result

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
    )


class DecisionRequest(ScoreRequest):
    operation: Literal["decision.score"] = "decision.score"
