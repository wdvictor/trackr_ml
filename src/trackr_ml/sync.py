from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from math import floor
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .config import Settings
from .domain import DatasetSpec, SyncResult, SyncState
from .storage import (
    SyncStateStore,
    append_notifications,
    load_csv_row_ids,
    utc_now_iso,
)

DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="financial",
        csv_filename="is_transactions_notifications.csv",
        isft=True,
    ),
    DatasetSpec(
        key="not_financial",
        csv_filename="is_not_financial_transaction.csv",
        isft=False,
    ),
    DatasetSpec(
        key="not_classified",
        csv_filename="not_classified.csv",
        isft=None,
    ),
)

HOLDOUT_RATIO = 0.2


def sanitize_api_url(api_url: str) -> str:
    split = urlsplit(api_url)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if key not in {"p", "size", "isft"}
    ]
    return urlunsplit(
        (split.scheme, split.netloc, split.path, urlencode(filtered_query), split.fragment)
    )


class NotificationsApiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.endpoint = sanitize_api_url(settings.api_url)

    def fetch_notifications(
        self, *, page: int, size: int, isft: bool | None
    ) -> list[dict[str, object]]:
        params: dict[str, object] = {"p": page, "size": size}
        if isft is not None:
            params["isft"] = "true" if isft else "false"

        response = self.session.get(
            self.endpoint,
            headers={"X-API-key": self.settings.api_key},
            params=params,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("A API retornou um payload diferente de uma lista.")

        return [self._normalize_record(item) for item in payload]

    @staticmethod
    def _normalize_record(payload: dict[str, object]) -> dict[str, object]:
        return {
            "id": int(payload["id"]),
            "app_name": str(payload.get("app_name", "") or ""),
            "text": str(payload.get("text", "") or ""),
            "is_financial_transaction": payload.get("is_financial_transaction"),
        }


class NotificationsSyncService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = NotificationsApiClient(settings)
        self.state_store = SyncStateStore(settings.cache_dir / "sync_state.json")

    def sync_all(self) -> list[SyncResult]:
        self.settings.ensure_directories()
        states = self.state_store.load()
        results: list[SyncResult] = []

        for spec in DATASET_SPECS:
            current_state = states.get(spec.key, SyncState())
            result, next_state = self._sync_dataset(spec, current_state)
            states[spec.key] = next_state
            results.append(result)

        self.state_store.save(states)
        return results

    def _sync_dataset(
        self, spec: DatasetSpec, current_state: SyncState
    ) -> tuple[SyncResult, SyncState]:
        page = 1
        highest_synced_id = current_state.highest_synced_id
        terminal_page_last_id = current_state.terminal_page_last_id
        last_page_synced = current_state.last_page_synced
        holdout_remainder = current_state.holdout_remainder
        collected: list[dict[str, object]] = []

        while True:
            batch = self.client.fetch_notifications(
                page=page, size=self.settings.page_size, isft=spec.isft
            )

            if not batch:
                last_page_synced = max(last_page_synced, page - 1)
                break

            batch_ids = [int(record["id"]) for record in batch]
            last_page_synced = page
            terminal_page_last_id = batch_ids[-1]
            collected.extend(self._filter_new_records(batch, highest_synced_id))

            if len(batch) < self.settings.page_size:
                break

            if (
                highest_synced_id
                and self._page_is_descending(batch_ids)
                and max(batch_ids) <= highest_synced_id
            ):
                break

            page += 1

        deduplicated = self._deduplicate_by_id(collected)
        raw_csv_path = self.settings.raw_data_dir / spec.csv_filename
        test_csv_path = (
            self.settings.test_data_dir / spec.csv_filename
            if spec.isft is not None
            else None
        )
        persisted_ids = self._load_persisted_ids(spec)
        new_records = [
            record for record in deduplicated if int(record["id"]) not in persisted_ids
        ]
        raw_records, test_records, holdout_remainder = self._split_records_for_storage(
            spec,
            new_records,
            holdout_remainder=holdout_remainder,
        )
        raw_records_written = append_notifications(raw_csv_path, raw_records)
        test_records_written = (
            append_notifications(test_csv_path, test_records)
            if test_csv_path is not None
            else 0
        )
        records_written = raw_records_written + test_records_written
        if deduplicated:
            highest_synced_id = max(
                highest_synced_id,
                max(int(record["id"]) for record in deduplicated),
            )

        next_state = SyncState(
            highest_synced_id=highest_synced_id,
            terminal_page_last_id=terminal_page_last_id,
            last_page_synced=last_page_synced,
            holdout_remainder=holdout_remainder,
            updated_at=utc_now_iso(),
        )

        return (
            SyncResult(
                dataset=spec.key,
                csv_path=str(raw_csv_path),
                records_written=records_written,
                highest_synced_id=highest_synced_id,
                terminal_page_last_id=terminal_page_last_id,
                last_page_synced=last_page_synced,
                raw_records_written=raw_records_written,
                test_csv_path=(
                    None if test_csv_path is None else str(test_csv_path)
                ),
                test_records_written=test_records_written,
            ),
            next_state,
        )

    @staticmethod
    def _filter_new_records(
        records: Iterable[dict[str, object]], highest_synced_id: int
    ) -> list[dict[str, object]]:
        if highest_synced_id <= 0:
            return list(records)

        return [
            record for record in records if int(record["id"]) > highest_synced_id
        ]

    @staticmethod
    def _deduplicate_by_id(
        records: Iterable[dict[str, object]]
    ) -> list[dict[str, object]]:
        deduplicated: dict[int, dict[str, object]] = {}
        for record in records:
            deduplicated[int(record["id"])] = record
        return sorted(deduplicated.values(), key=lambda item: int(item["id"]))

    def _load_persisted_ids(self, spec: DatasetSpec) -> set[int]:
        persisted_ids = load_csv_row_ids(self.settings.raw_data_dir / spec.csv_filename)
        if spec.isft is not None:
            persisted_ids.update(
                load_csv_row_ids(self.settings.test_data_dir / spec.csv_filename)
            )
        return persisted_ids

    @staticmethod
    def _holdout_target_count(
        record_count: int, holdout_remainder: float
    ) -> tuple[int, float]:
        desired_holdout = (record_count * HOLDOUT_RATIO) + holdout_remainder
        holdout_count = floor(desired_holdout)
        return holdout_count, desired_holdout - holdout_count

    @staticmethod
    def _select_holdout_ids(
        spec: DatasetSpec, records: Iterable[dict[str, object]], holdout_count: int
    ) -> set[int]:
        ranked_records = sorted(
            records,
            key=lambda item: (
                sha256(f"{spec.key}:{int(item['id'])}".encode("utf-8")).hexdigest(),
                int(item["id"]),
            ),
        )
        return {int(record["id"]) for record in ranked_records[:holdout_count]}

    def _split_records_for_storage(
        self,
        spec: DatasetSpec,
        records: list[dict[str, object]],
        *,
        holdout_remainder: float,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
        if spec.isft is None:
            return records, [], 0.0

        if not records:
            return [], [], holdout_remainder

        holdout_count, next_remainder = self._holdout_target_count(
            len(records),
            holdout_remainder,
        )
        if holdout_count <= 0:
            return records, [], next_remainder

        holdout_ids = self._select_holdout_ids(spec, records, holdout_count)
        raw_records: list[dict[str, object]] = []
        test_records: list[dict[str, object]] = []

        for record in records:
            if int(record["id"]) in holdout_ids:
                test_records.append(record)
            else:
                raw_records.append(record)

        return raw_records, test_records, next_remainder

    @staticmethod
    def _page_is_descending(batch_ids: list[int]) -> bool:
        if len(batch_ids) < 2:
            return True
        return batch_ids[0] >= batch_ids[-1]
