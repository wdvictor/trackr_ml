from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import Settings
from .model_registry import (
    build_model_paths,
    normalize_model_version,
    update_evaluation_report,
)
from .predictor import NotificationClassifier
from .storage import utc_now_iso
from .training import compute_abstention_metrics, compute_binary_metrics, load_labeled_examples


def evaluate_model(
    settings: Settings,
    *,
    version: str | None = None,
    model_path: Path | None = None,
) -> dict[str, object]:
    settings.ensure_directories()
    classifier = NotificationClassifier.load(
        model_path=model_path,
        model_version=version,
    )
    X, y = load_labeled_examples(settings)

    probabilities = [float(item[1]) for item in classifier.model.predict_proba(X)]
    binary_predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    thresholds = classifier.metadata["thresholds"]
    lower_bound = float(thresholds["unknown_lower_bound"])
    upper_bound = float(thresholds["unknown_upper_bound"])
    ternary_predictions = [
        (
            "financial_transaction"
            if probability >= upper_bound
            else (
                "not_financial_transaction"
                if probability <= lower_bound
                else "unknown"
            )
        )
        for probability in probabilities
    ]
    evaluated_at = utc_now_iso()

    report = {
        "evaluated_at": evaluated_at,
        "model": classifier.metadata.get("model"),
        "dataset": {
            "rows": len(X),
            "positive_rows": sum(1 for item in y if item == 1),
            "negative_rows": sum(1 for item in y if item == 0),
        },
        "binary": compute_binary_metrics(y, binary_predictions),
        "abstention": compute_abstention_metrics(
            probabilities,
            y,
            lower_bound,
            upper_bound,
        ),
        "ternary_predictions": dict(Counter(ternary_predictions)),
    }

    if version is None and classifier.metadata.get("model"):
        raw_version = classifier.metadata["model"].get("version")
        version = str(raw_version) if raw_version else None

    if version is not None:
        resolved_version = normalize_model_version(version)
        resolved_paths = build_model_paths(settings.models_dir, resolved_version)
        evaluation_path = resolved_paths["evaluation_path"]
        evaluation_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        update_evaluation_report(
            resolved_paths["registry_path"],
            version=resolved_version,
            evaluation_path=evaluation_path,
            evaluated_at=evaluated_at,
        )
        report["evaluation_path"] = str(evaluation_path)
    elif model_path is not None:
        evaluation_path = model_path.with_suffix(".evaluation.json")
        evaluation_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        report["evaluation_path"] = str(evaluation_path)

    return report
