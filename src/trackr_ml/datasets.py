from __future__ import annotations

from collections import Counter
from pathlib import Path

from .storage import load_csv_rows

LABELED_DATASET_FILENAMES: tuple[tuple[int, str], ...] = (
    (1, "is_transactions_notifications.csv"),
    (0, "is_not_financial_transaction.csv"),
)


def _parse_row_id(row: dict[str, str], source_path: Path) -> int:
    raw_id = (row.get("id", "") or "").strip()
    if not raw_id:
        raise RuntimeError(f"O arquivo {source_path} contem uma linha sem id.")

    try:
        return int(raw_id)
    except ValueError as exc:
        raise RuntimeError(
            f"O arquivo {source_path} contem um id invalido: {raw_id!r}."
        ) from exc


def load_labeled_row_ids(data_dir: Path) -> set[int]:
    row_ids: set[int] = set()

    for _, filename in LABELED_DATASET_FILENAMES:
        csv_path = data_dir / filename
        for row in load_csv_rows(csv_path):
            row_ids.add(_parse_row_id(row, csv_path))

    return row_ids


def load_labeled_examples(
    data_dir: Path,
    *,
    excluded_ids: set[int] | None = None,
    missing_data_message: str,
    insufficient_classes_message: str,
) -> tuple[list[dict[str, str]], list[int]]:
    skipped_ids = excluded_ids or set()
    X: list[dict[str, str]] = []
    y: list[int] = []

    for label, filename in LABELED_DATASET_FILENAMES:
        csv_path = data_dir / filename
        rows = load_csv_rows(csv_path)
        for row in rows:
            row_id = _parse_row_id(row, csv_path)
            if row_id in skipped_ids:
                continue

            text = (row.get("text", "") or "").strip()
            if not text:
                continue

            X.append(
                {
                    "text": text,
                    "app_name": row.get("app_name", "") or "",
                }
            )
            y.append(label)

    if not X:
        raise RuntimeError(missing_data_message)

    classes = Counter(y)
    if len(classes) < 2:
        raise RuntimeError(insufficient_classes_message)

    return X, y
