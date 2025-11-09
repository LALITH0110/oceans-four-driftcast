"""
File Summary:
- Collects schema validation and normalization helpers for crowdsourced data.
- Exposes high-level ``validate_payload`` and ``ingest_json`` functions.
- Integrates with driftcast.cli ``ingest`` command for future live data.
"""

from .schema import CrowdSchema, validate_payload
from .normalize import ingest_json_file

__all__ = ["CrowdSchema", "validate_payload", "ingest_json_file"]
