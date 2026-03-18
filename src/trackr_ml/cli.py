from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import run_predict
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trackr-ml",
        description=(
            "Sincroniza notificacoes, treina o modelo de classificacao e executa predicoes."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "sync",
        help="Baixa os dados do endpoint e atualiza os CSVs locais incrementalmente.",
    )

    subparsers.add_parser(
        "train",
        help="Treina o classificador com os CSVs rotulados e salva o modelo em models/.",
    )
    train_parser = subparsers.choices["train"]
    train_parser.add_argument(
        "--version",
        required=True,
        help="Versao do modelo. O artefato sera salvo como trackr-[versao].",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Executa sync seguido de train.",
    )
    pipeline_parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Treina apenas com os CSVs locais, sem consultar a API.",
    )
    pipeline_parser.add_argument(
        "--version",
        required=True,
        help="Versao do modelo. O artefato sera salvo como trackr-[versao].",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Classifica uma notificacao e extrai os dados da transacao quando aplicavel.",
    )
    predict_parser.add_argument("--text", required=True, help="Texto da notificacao.")
    predict_parser.add_argument(
        "--app-name",
        default="",
        help="Nome do app que gerou a notificacao.",
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Caminho alternativo para o arquivo .pkl do modelo.",
    )
    predict_parser.add_argument(
        "--model-version",
        default=None,
        help="Versao de um modelo registrado, por exemplo 1.0.0.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Testa um modelo salvo contra o dataset rotulado local e gera um relatorio.",
    )
    evaluate_parser.add_argument(
        "--version",
        default=None,
        help="Versao de um modelo registrado, por exemplo 1.0.0.",
    )
    evaluate_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Caminho alternativo para o arquivo .pkl do modelo.",
    )

    subparsers.add_parser(
        "list-models",
        help="Lista os modelos versionados registrados em models/registry.json.",
    )

    return parser


def run_sync() -> list[dict[str, object]]:
    from .sync import NotificationsSyncService

    settings = Settings.from_env(require_api=True)
    service = NotificationsSyncService(settings)
    return [result.to_dict() for result in service.sync_all()]


def run_train(version: str) -> dict[str, object]:
    from .training import train_model

    settings = Settings.from_env(require_api=False)
    return train_model(settings, version=version)


def run_pipeline(skip_sync: bool, version: str) -> dict[str, object]:
    payload: dict[str, object] = {}

    if not skip_sync:
        payload["sync"] = run_sync()
    else:
        payload["sync"] = "skipped"

    payload["train"] = run_train(version=version)
    return payload


def run_evaluate(
    version: str | None,
    model_path: Path | None,
) -> dict[str, object]:
    from .evaluation import evaluate_model

    if version is not None and model_path is not None:
        raise RuntimeError("Use apenas --model-path ou --version, nunca os dois.")

    settings = Settings.from_env(require_api=False)
    return evaluate_model(settings, version=version, model_path=model_path)


def run_list_models() -> list[dict[str, object]]:
    from .model_registry import list_registered_models

    settings = Settings.from_env(require_api=False)
    return list_registered_models(settings.models_dir)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "sync":
            payload = run_sync()
        elif args.command == "train":
            payload = run_train(version=args.version)
        elif args.command == "pipeline":
            payload = run_pipeline(skip_sync=args.skip_sync, version=args.version)
        elif args.command == "predict":
            payload = run_predict(
                text=args.text,
                app_name=args.app_name,
                model_path=args.model_path,
                model_version=args.model_version,
            )
        elif args.command == "evaluate":
            payload = run_evaluate(version=args.version, model_path=args.model_path)
        elif args.command == "list-models":
            payload = run_list_models()
        else:
            parser.error("Comando nao suportado.")
            return 2
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
