from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from .config import Settings
from .features import build_classifier_pipeline
from .model_registry import build_model_paths, register_model, sanitize_model_descriptor, to_relative_path_str
from .storage import load_csv_rows, utc_now_iso


def parse_isft(raw_value: str) -> bool | None:
    normalized = (raw_value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def load_labeled_examples(settings: Settings) -> tuple[list[dict[str, str]], list[int]]:
    financial_path = settings.raw_data_dir / "is_transactions_notifications.csv"
    not_financial_path = settings.raw_data_dir / "is_not_financial_transaction.csv"

    X: list[dict[str, str]] = []
    y: list[int] = []

    for label, path in ((1, financial_path), (0, not_financial_path)):
        rows = load_csv_rows(path)
        for row in rows:
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
        raise RuntimeError(
            "Nenhum dado rotulado foi encontrado. Execute o comando de sync antes do train."
        )

    classes = Counter(y)
    if len(classes) < 2:
        raise RuntimeError(
            "O treinamento precisa de exemplos das classes true e false."
        )

    return X, y


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def compute_binary_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, len(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_abstention_metrics(
    probabilities: list[float],
    y_true: list[int],
    lower_bound: float,
    upper_bound: float,
) -> dict[str, float]:
    covered = 0
    correct = 0
    unknown = 0

    for truth, probability in zip(y_true, probabilities):
        if probability >= upper_bound:
            prediction = 1
        elif probability <= lower_bound:
            prediction = 0
        else:
            unknown += 1
            continue

        covered += 1
        if prediction == truth:
            correct += 1

    return {
        "coverage": safe_divide(covered, len(y_true)),
        "selective_accuracy": safe_divide(correct, covered),
        "unknown_rate": safe_divide(unknown, len(y_true)),
    }


def train_model(
    settings: Settings,
    version: str,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    from sklearn.model_selection import train_test_split

    settings.ensure_directories()
    X, y = load_labeled_examples(settings)
    model = build_classifier_pipeline()

    class_counts = Counter(y)
    can_split = min(class_counts.values()) >= 2 and len(X) >= 10
    validation_metrics: dict[str, object]

    if can_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model.fit(X_train, y_train)
        probabilities = [float(item[1]) for item in model.predict_proba(X_val)]
        predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
        validation_metrics = {
            "binary": compute_binary_metrics(y_val, predictions),
            "abstention": compute_abstention_metrics(
                probabilities,
                y_val,
                settings.unknown_lower_bound,
                settings.unknown_upper_bound,
            ),
            "validation_rows": len(y_val),
        }
    else:
        model.fit(X, y)
        validation_metrics = {
            "binary": None,
            "abstention": None,
            "validation_rows": 0,
            "warning": "Dataset pequeno demais para holdout estratificado; modelo treinado com todos os dados.",
        }

    final_model = build_classifier_pipeline()
    final_model.fit(X, y)

    resolved_paths = build_model_paths(settings.models_dir, version)
    resolved_model_path = model_path or resolved_paths["model_path"]
    resolved_metadata_path = metadata_path or resolved_paths["metadata_path"]
    relative_model_path = to_relative_path_str(resolved_model_path)
    relative_metadata_path = to_relative_path_str(resolved_metadata_path)

    metadata = {
        "model": sanitize_model_descriptor(
            {
            "version": resolved_paths["version"],
            "name": resolved_paths["name"],
            "artifact_path": relative_model_path,
            "metadata_path": relative_metadata_path,
            }
        ),
        "trained_at": utc_now_iso(),
        "features": {
            "text": "tfidf_word_1_2grams + tfidf_char_3_5grams",
            "app_name": "one_hot_encoding",
        },
        "labels": {
            "positive": "financial_transaction",
            "negative": "not_financial_transaction",
            "uncertain": "unknown",
        },
        "thresholds": {
            "unknown_lower_bound": settings.unknown_lower_bound,
            "unknown_upper_bound": settings.unknown_upper_bound,
        },
        "dataset": {
            "rows": len(X),
            "positive_rows": class_counts[1],
            "negative_rows": class_counts[0],
        },
        "validation": validation_metrics,
    }

    artifact = {"model": final_model, "metadata": metadata}

    with resolved_model_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    resolved_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    register_model(
        resolved_paths["registry_path"],
        version=str(resolved_paths["version"]),
        model_name=str(resolved_paths["name"]),
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
        trained_at=str(metadata["trained_at"]),
        dataset_rows=len(X),
    )

    return {
        "model_name": str(resolved_paths["name"]),
        "model_version": str(resolved_paths["version"]),
        "model_path": relative_model_path,
        "metadata_path": relative_metadata_path,
        "metadata": metadata,
    }
