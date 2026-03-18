from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .domain import SyncState

CSV_FIELDNAMES = ["id", "app_name", "text", "is_financial_transaction"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_isft(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def append_notifications(csv_path: Path, records: Iterable[dict[str, object]]) -> int:
    materialized = list(records)
    if not materialized:
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for record in materialized:
            writer.writerow(
                {
                    "id": int(record["id"]),
                    "app_name": str(record.get("app_name", "") or ""),
                    "text": str(record.get("text", "") or ""),
                    "is_financial_transaction": serialize_isft(
                        record.get("is_financial_transaction")
                    ),
                }
            )

    return len(materialized)


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


class SyncStateStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load(self) -> dict[str, SyncState]:
        if not self.state_path.exists():
            return {}

        raw_payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            dataset_key: SyncState.from_dict(dataset_state)
            for dataset_key, dataset_state in raw_payload.items()
        }

    def save(self, states: dict[str, SyncState]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            dataset_key: state.to_dict() for dataset_key, state in states.items()
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
