from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from trackr_ml.datasets import load_labeled_examples, load_labeled_row_ids

CSV_FIELDNAMES = ["id", "app_name", "text", "is_financial_transaction"]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class DatasetLoadingTests(unittest.TestCase):
    def test_load_labeled_row_ids_reads_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_csv(
                data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "101",
                        "app_name": "com.bank",
                        "text": "Compra aprovada",
                        "is_financial_transaction": "true",
                    }
                ],
            )
            write_csv(
                data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "202",
                        "app_name": "com.chat",
                        "text": "Mensagem recebida",
                        "is_financial_transaction": "false",
                    }
                ],
            )

            self.assertEqual(load_labeled_row_ids(data_dir), {101, 202})

    def test_load_labeled_examples_excludes_ids_from_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_csv(
                data_dir / "is_transactions_notifications.csv",
                [
                    {
                        "id": "1",
                        "app_name": "com.bank",
                        "text": "Compra aprovada",
                        "is_financial_transaction": "true",
                    },
                    {
                        "id": "2",
                        "app_name": "com.bank",
                        "text": "Pix recebido",
                        "is_financial_transaction": "true",
                    },
                ],
            )
            write_csv(
                data_dir / "is_not_financial_transaction.csv",
                [
                    {
                        "id": "3",
                        "app_name": "com.chat",
                        "text": "Mensagem recebida",
                        "is_financial_transaction": "false",
                    }
                ],
            )

            X, y = load_labeled_examples(
                data_dir,
                excluded_ids={2},
                missing_data_message="missing",
                insufficient_classes_message="missing class",
            )

            self.assertEqual(
                X,
                [
                    {"text": "Compra aprovada", "app_name": "com.bank"},
                    {"text": "Mensagem recebida", "app_name": "com.chat"},
                ],
            )
            self.assertEqual(y, [1, 0])


if __name__ == "__main__":
    unittest.main()
