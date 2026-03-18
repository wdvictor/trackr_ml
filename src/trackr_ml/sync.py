from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .config import Settings
from .domain import DatasetSpec, SyncResult, SyncState
from .storage import SyncStateStore, append_notifications, utc_now_iso

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
        csv_path = self.settings.raw_data_dir / spec.csv_filename
        records_written = append_notifications(csv_path, deduplicated)
        if deduplicated:
            highest_synced_id = max(
                highest_synced_id,
                max(int(record["id"]) for record in deduplicated),
            )

        next_state = SyncState(
            highest_synced_id=highest_synced_id,
            terminal_page_last_id=terminal_page_last_id,
            last_page_synced=last_page_synced,
            updated_at=utc_now_iso(),
        )

        return (
            SyncResult(
                dataset=spec.key,
                csv_path=str(csv_path),
                records_written=records_written,
                highest_synced_id=highest_synced_id,
                terminal_page_last_id=terminal_page_last_id,
                last_page_synced=last_page_synced,
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

    @staticmethod
    def _page_is_descending(batch_ids: list[int]) -> bool:
        if len(batch_ids) < 2:
            return True
        return batch_ids[0] >= batch_ids[-1]
