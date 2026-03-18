from __future__ import annotations

from pathlib import Path

from .config import Settings
from .predictor import NotificationClassifier


def run_predict(
    text: str,
    app_name: str = "",
    model_path: str | Path | None = None,
    model_version: str | None = None,
) -> dict[str, object]:
    """Run a prediction using a specific model or the latest registered one."""

    if model_path is not None and model_version is not None:
        raise RuntimeError("Use apenas model_path ou model_version, nunca os dois.")

    Settings.from_env(require_api=False)
    classifier = NotificationClassifier.load(
        model_path=Path(model_path) if model_path is not None else None,
        model_version=model_version,
    )
    return classifier.predict(text=text, app_name=app_name).to_dict()
