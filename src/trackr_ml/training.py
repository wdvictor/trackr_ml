from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from .config import Settings
from .datasets import load_labeled_examples, load_labeled_row_ids
from .features import build_classifier_pipeline
from .metrics import compute_abstention_metrics, compute_binary_metrics
from .model_registry import build_model_paths, register_model, sanitize_model_descriptor, to_relative_path_str
from .storage import utc_now_iso


def train_model(
    settings: Settings,
    version: str,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    from sklearn.model_selection import train_test_split

    settings.ensure_directories()
    isolated_test_ids = load_labeled_row_ids(settings.test_data_dir)
    X, y = load_labeled_examples(
        settings.raw_data_dir,
        excluded_ids=isolated_test_ids,
        missing_data_message=(
            "Nenhum dado rotulado de treino foi encontrado. Execute o comando sync "
            "antes do train e garanta que o dataset isolado de teste nao consuma "
            "todos os exemplos."
        ),
        insufficient_classes_message=(
            "O treinamento precisa de exemplos de treino das classes true e false."
        ),
    )
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
