from __future__ import annotations

import json
import re
from pathlib import Path

MODEL_NAME_PREFIX = "trackr-"
MODEL_REGISTRY_FILENAME = "registry.json"
MODEL_FILE_SUFFIX = ".pkl"
METADATA_FILE_SUFFIX = ".json"
EVALUATION_FILE_SUFFIX = ".evaluation.json"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def normalize_model_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith(MODEL_NAME_PREFIX):
        normalized = normalized[len(MODEL_NAME_PREFIX) :]

    if not normalized or not VERSION_PATTERN.fullmatch(normalized):
        raise RuntimeError(
            "Versao invalida. Use apenas letras, numeros, ponto, underscore ou hifen."
        )

    return normalized


def build_model_name(version: str) -> str:
    return f"{MODEL_NAME_PREFIX}{normalize_model_version(version)}"


def build_model_paths(models_dir: Path, version: str) -> dict[str, Path | str]:
    normalized_version = normalize_model_version(version)
    model_name = build_model_name(normalized_version)
    return {
        "version": normalized_version,
        "name": model_name,
        "model_path": models_dir / f"{model_name}{MODEL_FILE_SUFFIX}",
        "metadata_path": models_dir / f"{model_name}{METADATA_FILE_SUFFIX}",
        "evaluation_path": models_dir / f"{model_name}{EVALUATION_FILE_SUFFIX}",
        "registry_path": models_dir / MODEL_REGISTRY_FILENAME,
    }


def load_registry(registry_path: Path) -> dict[str, object]:
    if not registry_path.exists():
        return {"latest_version": None, "models": {}}

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if "models" not in payload or not isinstance(payload["models"], dict):
        payload["models"] = {}
    payload.setdefault("latest_version", None)
    return payload


def save_registry(registry_path: Path, registry: dict[str, object]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def register_model(
    registry_path: Path,
    *,
    version: str,
    model_name: str,
    model_path: Path,
    metadata_path: Path,
    trained_at: str,
    dataset_rows: int,
    overwrite: bool = False,
) -> dict[str, object]:
    normalized_version = normalize_model_version(version)
    registry = load_registry(registry_path)
    models = registry["models"]

    if normalized_version in models and not overwrite:
        raise RuntimeError(
            f"O modelo {build_model_name(normalized_version)} ja existe. Use outra versao."
        )

    models[normalized_version] = {
        "version": normalized_version,
        "name": model_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "trained_at": trained_at,
        "dataset_rows": dataset_rows,
    }
    registry["latest_version"] = normalized_version
    save_registry(registry_path, registry)
    return models[normalized_version]


def update_evaluation_report(
    registry_path: Path,
    *,
    version: str,
    evaluation_path: Path,
    evaluated_at: str,
) -> None:
    normalized_version = normalize_model_version(version)
    registry = load_registry(registry_path)
    models = registry["models"]
    if normalized_version not in models:
        return

    models[normalized_version]["evaluation_path"] = str(evaluation_path)
    models[normalized_version]["last_evaluated_at"] = evaluated_at
    save_registry(registry_path, registry)


def resolve_registered_model(
    models_dir: Path, version: str | None = None
) -> dict[str, object]:
    registry_path = models_dir / MODEL_REGISTRY_FILENAME
    registry = load_registry(registry_path)
    models = registry["models"]

    target_version = (
        normalize_model_version(version)
        if version is not None
        else registry.get("latest_version")
    )
    if not target_version:
        raise RuntimeError("Nenhum modelo versionado foi encontrado em models/.")

    model_info = models.get(target_version)
    if not isinstance(model_info, dict):
        raise RuntimeError(
            f"O modelo {build_model_name(target_version)} nao esta registrado."
        )

    return model_info


def list_registered_models(models_dir: Path) -> list[dict[str, object]]:
    registry_path = models_dir / MODEL_REGISTRY_FILENAME
    registry = load_registry(registry_path)
    models = registry["models"]
    ordered_versions = sorted(
        models.keys(),
        key=lambda item: (
            models[item].get("trained_at", ""),
            item,
        ),
    )
    return [models[version] for version in reversed(ordered_versions)]
