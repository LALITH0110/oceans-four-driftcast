# Copyright (c) 2025 Oceans Four Driftcast Team
# SPDX-License-Identifier: MIT
"""
File Summary:
- Normalizes validated crowdsourced drifter JSON into tabular Parquet storage.
- Provides CLI-friendly helper that reads JSON, validates, and writes parquet files.
- Cooperates with tests to demonstrate ingest of mocked community observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from driftcast import logger

from .schema import CrowdSchema, validate_many


def _load_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, list) else [payload]


def _qa_checks(frame: pd.DataFrame) -> None:
    if not frame["notes"].fillna("").astype(str).str.strip().all():
        raise ValueError("QA validation failed: all observations require non-empty notes.")
    if not frame["lat"].astype(float).between(-90.0, 90.0).all():
        raise ValueError("QA validation failed: latitude values must be within [-90, 90].")
    if not frame["lon"].astype(float).between(-180.0, 180.0).all():
        raise ValueError("QA validation failed: longitude values must be within [-180, 180].")


def _apply_deduplication(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "item_type" in frame.columns:
        item_column = "item_type"
    elif "observed_item_type" in frame.columns:
        item_column = "observed_item_type"
    else:
        raise KeyError("Expected 'item_type' or 'observed_item_type' column for deduplication.")

    key = list(
        zip(
            frame["timestamp"].dt.floor("s"),
            frame["lat"].round(3),
            frame["lon"].round(3),
            frame[item_column].astype(str).str.lower(),
        )
    )
    frame = frame.assign(_dup_key=key)
    deduped = frame.drop_duplicates("_dup_key").drop(columns="_dup_key")
    removed = len(frame) - len(deduped)
    return deduped, removed


def _normalize_records(records: Iterable[dict], source_name: str) -> Tuple[pd.DataFrame, int]:
    frame = pd.DataFrame.from_records(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["est_size_mm"] = frame["est_size_mm"].astype(float)
    frame["source_file"] = source_name
    _qa_checks(frame)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame, removed = _apply_deduplication(frame)
    if removed:
        logger.warning("Duplicate crowd observations removed: %d entries", removed)
    return frame, removed


def ingest_json_file(
    json_path: Path | str,
    schema: CrowdSchema,
    output_dir: Path | str = Path("data/crowd/processed"),
) -> Path:
    """Ingest a JSON file, returning the path of the written Parquet dataset."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(json_path)
    validate_many(records, schema)
    table, removed = _normalize_records(records, source_name=json_path.name)
    table["date"] = table["timestamp"].dt.date.astype(str)
    dataset_root = output_dir / json_path.stem
    dataset_root.mkdir(parents=True, exist_ok=True)
    table.to_parquet(dataset_root, index=False, partition_cols=["date"])
    logger.info(
        "Ingested %d crowd observations (partitioned by date, removed %d duplicates) -> %s",
        len(table),
        removed,
        dataset_root,
    )
    return dataset_root

def validate_json_file(json_path: Path | str, schema: CrowdSchema) -> Tuple[int, int]:
    """Validate a JSON payload without writing outputs; returns (records, duplicates_removed)."""
    json_path = Path(json_path)
    records = _load_records(json_path)
    validate_many(records, schema)
    frame, removed = _normalize_records(records, source_name=json_path.name)
    return len(frame), removed
