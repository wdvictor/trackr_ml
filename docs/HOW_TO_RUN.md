# How to Run the Project

This guide shows the step-by-step flow to:

- set up the environment
- fetch data from the API
- train a new model
- test a trained model
- run predictions

## 1. Prerequisites

You need:

- Python 3.12 or newer
- `uv` installed
- access to the `NOTIFICATIONS_API_URL` value
- access to the `NOTIFICATIONS_API_KEY` value

If you prefer, the project also works without installing the package by using `PYTHONPATH=src`.

## 2. Set Up the Environment

1. Create the virtual environment:

```bash
uv venv
```

2. Activate the virtual environment:

```bash
. .venv/bin/activate
```

3. Install the project dependencies:

```bash
uv pip install -e .
```

## 3. Configure the `.env`

1. Copy the example file:

```bash
cp .env.example .env
```

2. Fill `.env` with the real values:

```dotenv
NOTIFICATIONS_API_URL=
NOTIFICATIONS_API_KEY=
UNKNOWN_LOWER_BOUND=0.35
UNKNOWN_UPPER_BOUND=0.65
REQUEST_TIMEOUT_SECONDS=30
```

Fields:

- `NOTIFICATIONS_API_URL`: base endpoint used by the sync command
- `NOTIFICATIONS_API_KEY`: value sent in the `X-API-key` header
- `UNKNOWN_LOWER_BOUND`: lower bound for `not_financial_transaction`
- `UNKNOWN_UPPER_BOUND`: upper bound for `financial_transaction`
- `REQUEST_TIMEOUT_SECONDS`: timeout for HTTP requests

Notes:

- `.env` must not be versioned
- the project reads `NOTIFICATIONS_API_URL` and `NOTIFICATIONS_API_KEY` directly from `.env`

## 4. How to Fetch Data from the API

To download the data and generate the local CSV files, run:

```bash
PYTHONPATH=src python -m trackr_ml.cli sync
```

This command collects all 3 datasets:

- notifications with `isft=true`
- notifications with `isft=false`
- unclassified notifications

Generated files:

- `data/raw/is_transactions_notifications.csv`
- `data/raw/is_not_financial_transaction.csv`
- `data/raw/not_classified.csv`

Incremental cache:

- `data/cache/sync_state.json`

The `sync` command already handles:

- sending `X-API-key` in the header
- using `NOTIFICATIONS_API_URL` from `.env`
- automatic pagination with `p`
- fetching up to `size=2000` per request
- avoiding duplicate `id` values
- continuing incrementally across executions

When to use it:

- run it before training a new model
- run it again whenever you want to refresh the local CSVs with new data

## 5. How to Train a New Model

Every model requires an explicit version.

Example:

```bash
PYTHONPATH=src python -m trackr_ml.cli train --version 1.0.0
```

Generated model name:

- `trackr-1.0.0`

Generated artifacts:

- `models/trackr-1.0.0.pkl`
- `models/trackr-1.0.0.json`
- `models/registry.json`

Training uses:

- `data/raw/is_transactions_notifications.csv`
- `data/raw/is_not_financial_transaction.csv`

Recommended flow for a new training run:

1. Refresh the data with `sync`
2. Prepare the isolated test dataset
3. Choose a new version
4. Run `train --version <new-version>`
5. Run `evaluate --version <new-version>`

## 6. How to Prepare an Isolated Test Dataset

Model accuracy must be measured on a holdout dataset that is never reused for training.

Create these files:

- `data/test/is_transactions_notifications.csv`
- `data/test/is_not_financial_transaction.csv`

Use the same CSV schema as the synced labeled files:

- `id`
- `app_name`
- `text`
- `is_financial_transaction`

Requirements for this dataset:

- it must contain examples that are not used for training
- it must contain both labels
- it should be representative of real production traffic
- it should remain stable across model comparisons
- if it was carved out from the same original source, keep the same `id` values

Important note:

- training reads `data/raw/`, but it now excludes any row whose `id` appears in `data/test/`
- this prevents data leakage when the holdout dataset was separated from the same synced pool

## 7. How to Test a New Model

After training, test the saved model with:

```bash
PYTHONPATH=src python -m trackr_ml.cli evaluate --version 1.0.0
```

This command:

- loads the versioned model saved in `models/`
- uses the isolated CSVs in `data/test/` as the evaluation dataset
- computes binary metrics
- computes metrics for the `unknown` strategy
- saves a report in `models/trackr-1.0.0.evaluation.json`
- updates `models/registry.json` with the latest evaluation

If you want to test an artifact outside the registry:

```bash
PYTHONPATH=src python -m trackr_ml.cli evaluate --model-path /path/to/model.pkl
```

Important note:

- this test uses only the holdout dataset in `data/test/`
- those examples must not be reused for training or model tuning

## 8. How to List Versioned Models

To see all registered models:

```bash
PYTHONPATH=src python -m trackr_ml.cli list-models
```

This command shows:

- version
- model name
- `.pkl` path
- metadata path
- training timestamp
- latest evaluation report, when available

For security, paths stored in `models/registry.json` use repository-relative paths instead of absolute OS paths.

## 9. How to Run the Full Pipeline

If you want to sync the data and then train:

```bash
PYTHONPATH=src python -m trackr_ml.cli pipeline --version 1.0.0
```

If the CSV files are already up to date and you want to skip the API:

```bash
PYTHONPATH=src python -m trackr_ml.cli pipeline --skip-sync --version 1.0.0
```

## 10. How to Run a Manual Prediction

To classify a notification with a versioned model:

```bash
PYTHONPATH=src python -m trackr_ml.cli predict \
  --model-version 1.0.0 \
  --app-name com.nu.production \
  --text "Compra aprovada no credito final 1234 no valor de R$ 55,90"
```

If `--model-version` is omitted, the project tries to use the most recent versioned model registered in `models/registry.json`.

## 11. Important File Structure

Configuration files:

- `.env`
- `.env.example`
- `pyproject.toml`

Data:

- `data/raw/`
- `data/test/`
- `data/cache/`

Models:

- `models/*.pkl`
- `models/*.json`

Main code:

- `src/trackr_ml/cli.py`
- `src/trackr_ml/sync.py`
- `src/trackr_ml/training.py`
- `src/trackr_ml/evaluation.py`
- `src/trackr_ml/predictor.py`

## 12. Troubleshooting

Credential or endpoint error:

- review `NOTIFICATIONS_API_URL` and `NOTIFICATIONS_API_KEY` in `.env`

Error saying there is no labeled data:

- run `PYTHONPATH=src python -m trackr_ml.cli sync`

Error saying there is no isolated test dataset:

- create `data/test/is_transactions_notifications.csv`
- create `data/test/is_not_financial_transaction.csv`
- make sure both files contain at least one labeled example and that the examples were not used for training

Error saying the model does not exist:

- check the version in `models/registry.json`
- run `PYTHONPATH=src python -m trackr_ml.cli list-models`

Duplicate version error:

- use a new version in `train`, for example `1.0.1`

## 13. Automated Project Tests

To run the unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## 14. Recommended Flow Summary

1. Configure `.env`
2. Install dependencies
3. Run `sync`
4. Prepare `data/test/` with isolated labeled examples
5. Run `train --version <version>`
6. Run `evaluate --version <version>`
7. If approved, use `predict --model-version <version>`
