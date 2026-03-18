from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trackr_ml.config import Settings
from trackr_ml.evaluation import evaluate_model

CSV_FIELDNAMES = ["id", "app_name", "text", "is_financial_transaction"]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_settings(root_dir: Path) -> Settings:
    return Settings(
        api_url="",
        api_key="",
        raw_data_dir=root_dir / "data" / "raw",
        test_data_dir=root_dir / "data" / "test",
        cache_dir=root_dir / "data" / "cache",
        models_dir=root_dir / "models",
        page_size=2000,
        request_timeout_seconds=30,
        unknown_lower_bound=0.35,
        unknown_upper_bound=0.65,
    )


class FakeModel:
    def __init__(self) -> None:
        self.seen_rows: list[dict[str, str]] = []

    def predict_proba(self, rows: list[dict[str, str]]) -> list[list[float]]:
        self.seen_rows = list(rows)
        return [
            [0.1, 0.9] if row["text"] == "test positive" else [0.9, 0.1]
            for row in rows
        ]


class EvaluationTests(unittest.TestCase):
    @patch("trackr_ml.evaluation.NotificationClassifier.load")
    def test_evaluate_model_uses_isolated_test_dataset(self, load_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            settings = build_settings(root_dir)
            fake_model = FakeModel()
            load_mock.return_value = SimpleNamespace(
                model=fake_model,
                metadata={
                    "thresholds": {
                        "unknown_lower_bound": 0.35,
                        "unknown_upper_bound": 0.65,
                    },
                    "model": None,
                },
            )

            write_csv(
                settings.raw_data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "1",
                        "app_name": "com.bank",
                        "text": "raw positive",
                        "is_financial_transaction": "true",
                    }
                ],
            )
            write_csv(
                settings.raw_data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "2",
                        "app_name": "com.chat",
                        "text": "raw negative",
                        "is_financial_transaction": "false",
                    }
                ],
            )
            write_csv(
                settings.test_data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "10",
                        "app_name": "com.bank",
                        "text": "test positive",
                        "is_financial_transaction": "true",
                    }
                ],
            )
            write_csv(
                settings.test_data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "11",
                        "app_name": "com.chat",
                        "text": "test negative",
                        "is_financial_transaction": "false",
                    }
                ],
            )

            model_path = settings.models_dir / "manual.pkl"
            report = evaluate_model(settings, model_path=model_path)

            self.assertEqual(
                fake_model.seen_rows,
                [
                    {"text": "test positive", "app_name": "com.bank"},
                    {"text": "test negative", "app_name": "com.chat"},
                ],
            )
            self.assertEqual(report["dataset"]["rows"], 2)
            self.assertEqual(report["dataset"]["positive_rows"], 1)
            self.assertEqual(report["dataset"]["negative_rows"], 1)
            self.assertTrue(model_path.with_suffix(".evaluation.json").exists())

    @patch("trackr_ml.evaluation.NotificationClassifier.load")
    def test_evaluate_model_requires_isolated_test_dataset(self, load_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = build_settings(Path(temp_dir))
            load_mock.return_value = SimpleNamespace(
                model=FakeModel(),
                metadata={
                    "thresholds": {
                        "unknown_lower_bound": 0.35,
                        "unknown_upper_bound": 0.65,
                    },
                    "model": None,
                },
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Nenhum dataset isolado de teste foi encontrado em data/test",
            ):
                evaluate_model(settings, model_path=settings.models_dir / "manual.pkl")


if __name__ == "__main__":
    unittest.main()
