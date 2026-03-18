from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DatasetKey = Literal["financial", "not_financial", "not_classified"]
PredictionLabel = Literal[
    "financial_transaction", "not_financial_transaction", "unknown"
]


@dataclass(slots=True)
class SyncState:
    highest_synced_id: int = 0
    terminal_page_last_id: int = 0
    last_page_synced: int = 0
    holdout_remainder: float = 0.0
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object] | None) -> "SyncState":
        if not raw:
            return cls()

        return cls(
            highest_synced_id=int(raw.get("highest_synced_id", 0) or 0),
            terminal_page_last_id=int(raw.get("terminal_page_last_id", 0) or 0),
            last_page_synced=int(raw.get("last_page_synced", 0) or 0),
            holdout_remainder=float(raw.get("holdout_remainder", 0.0) or 0.0),
            updated_at=(
                None if raw.get("updated_at") in (None, "") else str(raw["updated_at"])
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSpec:
    key: DatasetKey
    csv_filename: str
    isft: bool | None


@dataclass(slots=True)
class SyncResult:
    dataset: DatasetKey
    csv_path: str
    records_written: int
    highest_synced_id: int
    terminal_page_last_id: int
    last_page_synced: int
    raw_records_written: int = 0
    test_csv_path: str | None = None
    test_records_written: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TransactionDetails:
    value: float | None
    direction: Literal["income", "expense"] | None
    is_pix: bool
    card_type: Literal["credit", "debit"] | None
    card_last4: str | None
    card_label: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PredictionResult:
    label: PredictionLabel
    confidence: float
    transaction: TransactionDetails | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.transaction is None:
            payload["transaction"] = None
        return payload
