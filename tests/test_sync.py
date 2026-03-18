from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from trackr_ml.config import Settings
from trackr_ml.sync import NotificationsSyncService, sanitize_api_url

CSV_FIELDNAMES = ["id", "app_name", "text", "is_financial_transaction"]


class StubNotificationsClient:
    def __init__(
        self,
        payloads: dict[tuple[bool | None, int], list[dict[str, object]]],
    ) -> None:
        self.payloads = payloads

    def fetch_notifications(
        self, *, page: int, size: int, isft: bool | None
    ) -> list[dict[str, object]]:
        return [dict(item) for item in self.payloads.get((isft, page), [])]


def build_settings(root_dir: Path) -> Settings:
    return Settings(
        api_url="https://notifications.example/api/notifications",
        api_key="secret",
        raw_data_dir=root_dir / "data" / "raw",
        test_data_dir=root_dir / "data" / "test",
        cache_dir=root_dir / "data" / "cache",
        models_dir=root_dir / "models",
        page_size=50,
        request_timeout_seconds=30,
        unknown_lower_bound=0.35,
        unknown_upper_bound=0.65,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_ids(path: Path) -> list[int]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(row["id"]) for row in reader]


def build_records(
    start_id: int,
    end_id: int,
    *,
    app_name: str,
    text_prefix: str,
    isft: bool | None,
) -> list[dict[str, object]]:
    return [
        {
            "id": row_id,
            "app_name": app_name,
            "text": f"{text_prefix} {row_id}",
            "is_financial_transaction": isft,
        }
        for row_id in range(start_id, end_id + 1)
    ]


class SyncTests(unittest.TestCase):
    def test_sanitize_api_url_removes_managed_params(self) -> None:
        url = (
            "https://notifications.example/api/notifications"
            "?p=1&size=2000&isft=true&foo=bar"
        )

        self.assertEqual(
            sanitize_api_url(url),
            "https://notifications.example/api/notifications?foo=bar",
        )

    def test_sync_routes_twenty_percent_of_new_labeled_rows_to_test_without_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            settings = build_settings(root_dir)
            settings.ensure_directories()

            write_csv(
                settings.raw_data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "1",
                        "app_name": "com.bank",
                        "text": "Compra 1",
                        "is_financial_transaction": "true",
                    }
                ],
            )
            write_csv(
                settings.test_data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "2",
                        "app_name": "com.bank",
                        "text": "Compra 2",
                        "is_financial_transaction": "true",
                    }
                ],
            )
            write_csv(
                settings.raw_data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "101",
                        "app_name": "com.chat",
                        "text": "Mensagem 101",
                        "is_financial_transaction": "false",
                    }
                ],
            )
            write_csv(
                settings.test_data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "102",
                        "app_name": "com.chat",
                        "text": "Mensagem 102",
                        "is_financial_transaction": "false",
                    }
                ],
            )

            payloads = {
                (True, 1): build_records(
                    1,
                    12,
                    app_name="com.bank",
                    text_prefix="Compra",
                    isft=True,
                ),
                (False, 1): build_records(
                    101,
                    112,
                    app_name="com.chat",
                    text_prefix="Mensagem",
                    isft=False,
                ),
                (None, 1): build_records(
                    201,
                    203,
                    app_name="com.misc",
                    text_prefix="Notificacao",
                    isft=None,
                ),
            }

            first_service = NotificationsSyncService(settings)
            first_service.client = StubNotificationsClient(payloads)
            first_results = {
                result.dataset: result for result in first_service.sync_all()
            }

            financial_raw_ids = read_ids(
                settings.raw_data_dir / "is_transactions_notifications.csv"
            )
            financial_test_ids = read_ids(
                settings.test_data_dir / "is_transactions_notifications.csv"
            )
            self.assertEqual(len(financial_raw_ids), 9)
            self.assertEqual(len(financial_test_ids), 3)
            self.assertFalse(set(financial_raw_ids) & set(financial_test_ids))
            self.assertEqual(
                set(financial_raw_ids) | set(financial_test_ids),
                set(range(1, 13)),
            )

            not_financial_raw_ids = read_ids(
                settings.raw_data_dir / "is_not_financial_transaction.csv"
            )
            not_financial_test_ids = read_ids(
                settings.test_data_dir / "is_not_financial_transaction.csv"
            )
            self.assertEqual(len(not_financial_raw_ids), 9)
            self.assertEqual(len(not_financial_test_ids), 3)
            self.assertFalse(set(not_financial_raw_ids) & set(not_financial_test_ids))
            self.assertEqual(
                set(not_financial_raw_ids) | set(not_financial_test_ids),
                set(range(101, 113)),
            )

            self.assertEqual(
                read_ids(settings.raw_data_dir / "not_classified.csv"),
                [201, 202, 203],
            )
            self.assertFalse((settings.test_data_dir / "not_classified.csv").exists())

            self.assertEqual(first_results["financial"].raw_records_written, 8)
            self.assertEqual(first_results["financial"].test_records_written, 2)
            self.assertEqual(first_results["financial"].records_written, 10)
            self.assertEqual(first_results["not_financial"].raw_records_written, 8)
            self.assertEqual(first_results["not_financial"].test_records_written, 2)
            self.assertEqual(first_results["not_financial"].records_written, 10)
            self.assertEqual(first_results["not_classified"].raw_records_written, 3)
            self.assertEqual(first_results["not_classified"].test_records_written, 0)

            second_service = NotificationsSyncService(settings)
            second_service.client = StubNotificationsClient(payloads)
            second_results = {
                result.dataset: result for result in second_service.sync_all()
            }

            self.assertEqual(second_results["financial"].records_written, 0)
            self.assertEqual(second_results["not_financial"].records_written, 0)
            self.assertEqual(second_results["not_classified"].records_written, 0)
            self.assertEqual(
                read_ids(settings.raw_data_dir / "is_transactions_notifications.csv"),
                financial_raw_ids,
            )
            self.assertEqual(
                read_ids(settings.test_data_dir / "is_transactions_notifications.csv"),
                financial_test_ids,
            )
            self.assertEqual(
                read_ids(settings.raw_data_dir / "is_not_financial_transaction.csv"),
                not_financial_raw_ids,
            )
            self.assertEqual(
                read_ids(settings.test_data_dir / "is_not_financial_transaction.csv"),
                not_financial_test_ids,
            )


if __name__ == "__main__":
    unittest.main()
