from __future__ import annotations

import pickle
from pathlib import Path

from .config import ROOT_DIR
from .domain import PredictionLabel, PredictionResult
from .extraction import extract_transaction_details
from .model_registry import resolve_registered_model

DEFAULT_MODEL_FILENAME = "notification_classifier.pkl"
DEFAULT_MODEL_METADATA_FILENAME = "notification_classifier_metadata.json"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / DEFAULT_MODEL_FILENAME


def resolve_label(
    probability: float, lower_bound: float, upper_bound: float
) -> PredictionLabel:
    if probability >= upper_bound:
        return "financial_transaction"
    if probability <= lower_bound:
        return "not_financial_transaction"
    return "unknown"


class NotificationClassifier:
    def __init__(self, model, metadata: dict[str, object]) -> None:
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        model_version: str | None = None,
    ) -> "NotificationClassifier":
        if model_path is not None:
            resolved_path = model_path
        elif model_version is not None:
            resolved_path = Path(
                resolve_registered_model(ROOT_DIR / "models", model_version)["model_path"]
            )
        else:
            try:
                resolved_path = Path(
                    resolve_registered_model(ROOT_DIR / "models")["model_path"]
                )
            except RuntimeError:
                resolved_path = DEFAULT_MODEL_PATH

        if not resolved_path.exists():
            raise RuntimeError(
                f"Modelo nao encontrado em {resolved_path}. Execute o comando train."
            )

        with resolved_path.open("rb") as handle:
            artifact = pickle.load(handle)

        return cls(model=artifact["model"], metadata=artifact["metadata"])

    def predict(self, text: str, app_name: str = "") -> PredictionResult:
        if not text.strip():
            return PredictionResult(
                label="unknown",
                confidence=0.0,
                transaction=None,
            )

        probability = float(
            self.model.predict_proba([{"text": text, "app_name": app_name}])[0][1]
        )
        thresholds = self.metadata["thresholds"]
        lower_bound = float(thresholds["unknown_lower_bound"])
        upper_bound = float(thresholds["unknown_upper_bound"])
        label = resolve_label(probability, lower_bound, upper_bound)
        confidence = round(max(probability, 1 - probability), 4)

        return PredictionResult(
            label=label,
            confidence=confidence,
            transaction=(
                extract_transaction_details(text=text, app_name=app_name)
                if label == "financial_transaction"
                else None
            ),
        )
