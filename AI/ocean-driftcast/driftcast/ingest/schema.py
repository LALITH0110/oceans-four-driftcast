"""
File Summary:
- Loads and validates driftcast crowdsourced drifter observations using JSON schema.
- Wraps jsonschema library to provide reusable validation utilities.
- Schema lives in schemas/crowd_drifters.schema.json alongside repository docs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class CrowdSchema:
    """Lightweight container holding the compiled JSON schema validator."""

    path: Path
    validator: Draft202012Validator

    @classmethod
    def load(cls, path: Path | str) -> "CrowdSchema":
        """Load and compile a JSON schema from disk."""
        schema_path = Path(path)
        with schema_path.open("r", encoding="utf8") as fh:
            schema_obj = json.load(fh)
        validator = Draft202012Validator(schema_obj)
        return cls(path=schema_path, validator=validator)


def validate_payload(payload: Dict[str, Any], schema: CrowdSchema) -> None:
    """Validate a single JSON payload; raises ``jsonschema.ValidationError`` on failure."""
    schema.validator.validate(payload)


def validate_many(payloads: Iterable[Dict[str, Any]], schema: CrowdSchema) -> None:
    """Validate an iterable of payloads for bulk ingestion."""
    for item in payloads:
        validate_payload(item, schema)
