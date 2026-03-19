# trackr-ml

Step-by-step operational guide: [HOW_TO_RUN.md](./docs/HOW_TO_RUN.md)
Import guide for another Python project: [HOW_TO_IMPORT.md](./docs/HOW_TO_IMPORT.md)

Python pipeline to:

- sync notifications from the remote endpoint into local CSV files
- train a financial notification classifier
- version every trained model using the `trackr-[version]` pattern
- test a saved model against the local labeled dataset
- classify a new notification as `financial_transaction`, `not_financial_transaction`, or `unknown`
- extract transaction data when the notification is financial

## Architecture

### Where Machine Learning Is Used

Machine learning handles notification classification using:

- `text` with word and character `TF-IDF`
- `app_name` with `OneHotEncoder`
- `LogisticRegression` with `class_weight="balanced"`

`app_name` is kept as a feature because, in real notifications, the sender often provides strong contextual signals. Examples: banking apps, digital wallets, SMS apps, carriers, and the operating system.

### Where Procedural Code Is Used

After classification, transaction field extraction uses deterministic rules:

- `value`: regex for BRL currency values
- `direction`: heuristics for `income` or `expense`
- `isCompleted`: heuristics for completed, failed, or `unknow`
- `is_pix`: PIX-related keywords
- `card_type`: heuristics for `credit` or `debit`
- `card_last4` and `card_label`: best-effort card extraction

This split keeps the model focused on the semantic problem and keeps structured extraction explainable and easier to evolve.

### `unknown` Class

The labeled dataset exposed by the API contains `true` and `false`, but it does not provide a third labeled class for `unknown`. Because of that, the project implements `unknown` as confidence-based abstention:

- `financial_transaction` when the probability is >= `UNKNOWN_UPPER_BOUND`
- `not_financial_transaction` when the probability is <= `UNKNOWN_LOWER_BOUND`
- `unknown` in the middle zone

## Incremental Synchronization

Generated files:

- `data/raw/is_transactions_notifications.csv`
- `data/raw/is_not_financial_transaction.csv`
- `data/raw/not_classified.csv`
- `data/test/is_transactions_notifications.csv`
- `data/test/is_not_financial_transaction.csv`

Local cache:

- `data/cache/sync_state.json`

The cache stores:

- `highest_synced_id`: highest `id` safely persisted so far
- `terminal_page_last_id`: `id` of the last item received on the terminal page
- `last_page_synced`: last page visited

Important note: the project stores the page value as requested, but uses `highest_synced_id` as the real incremental reference. In APIs paginated with newest records first, resuming only from the last page can miss new data because page boundaries shift over time.

For labeled records, `sync` now routes roughly 20% of each new incremental batch into `data/test/` and keeps the remaining 80% in `data/raw/`. Duplicate `id` values are ignored across both directories.

## Model Versioning

Every training run now requires an explicit version.

Example:

```bash
PYTHONPATH=src python -m trackr_ml.cli train --version 1.0.0
```

Generated artifacts:

- `models/trackr-1.0.0.pkl`
- `models/trackr-1.0.0.json`
- `models/registry.json`

`registry.json` stores:

- registered versions
- the latest version
- repository-relative artifact paths
- training timestamp
- latest evaluation report, when available

## Holdout Test Dataset

Model accuracy must be measured on an isolated test dataset that is never reused for training.

Expected files:

- `data/test/is_transactions_notifications.csv`
- `data/test/is_not_financial_transaction.csv`

These files must follow the same CSV schema used by the synced labeled data:

- `id`
- `app_name`
- `text`
- `is_financial_transaction`

These files are populated automatically by `sync` with roughly 20% of new labeled rows, but they can also be curated manually if you need a frozen benchmark. Training still reads the labeled data from `data/raw/`, but it excludes any row whose `id` is present in `data/test/`. This prevents leakage when the holdout set was carved out from the same source dataset.

## Configuration

Copy `.env.example` to `.env` and fill it in:

```dotenv
NOTIFICATIONS_API_URL=
NOTIFICATIONS_API_KEY=replace-with-your-api-key
UNKNOWN_LOWER_BOUND=0.35
UNKNOWN_UPPER_BOUND=0.65
REQUEST_TIMEOUT_SECONDS=30
```

`.env` is ignored by Git.

## Installation

Using `uv`:

```bash
uv venv
. .venv/bin/activate
uv pip install -e .
```

Without installing the package, it also works with `PYTHONPATH=src`.

To consume this code from another Python project without publishing it, the recommended approach is a local editable install that points to this repository. That keeps the package importable as `trackr_ml` and preserves access to the local `models/` registry and artifacts stored here.

Example from the consumer project environment:

```bash
python -m pip install -e /absolute/path/to/trackr_ml
```

After that, the other project can import the public API with:

```python
from trackr_ml import run_predict
```

## Usage

Sync CSV files:

```bash
PYTHONPATH=src python -m trackr_ml.cli sync
```

Train the model:

```bash
PYTHONPATH=src python -m trackr_ml.cli train --version 1.0.0
```

Run the full pipeline:

```bash
PYTHONPATH=src python -m trackr_ml.cli pipeline --version 1.0.0
```

Classify a notification:

```bash
PYTHONPATH=src python -m trackr_ml.cli predict \
  --model-version 1.0.0 \
  --app-name com.nu.production \
  --text "Compra aprovada no credito final 1234 no valor de R$ 55,90"
```

If `--model-version` is omitted, prediction uses the most recent versioned model registered in `models/registry.json`.

## Library Usage

The supported public API for other Python projects is `run_predict`.

Example:

```python
from trackr_ml import run_predict

result = run_predict(
    text="Compra aprovada no credito final 1234 no valor de R$ 55,90",
    app_name="com.nu.production",
)
```

By default, `run_predict` uses the latest model registered in `models/registry.json`.

If needed, it can target a specific artifact:

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    app_name="com.picpay",
    model_version="1.0.0",
)
```

Or an explicit file path:

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    model_path="/absolute/path/to/model.pkl",
)
```

List versioned models:

```bash
PYTHONPATH=src python -m trackr_ml.cli list-models
```

## How to Test a Model

To test an already trained model:

```bash
PYTHONPATH=src python -m trackr_ml.cli evaluate --version 1.0.0
```

The command:

- loads `models/trackr-1.0.0.pkl`
- evaluates the model using the isolated labeled CSV files in `data/test/`
- computes binary metrics and `unknown`-class metrics
- saves the report to `models/trackr-1.0.0.evaluation.json`
- updates `models/registry.json` with the latest evaluation

It is also possible to test an artifact outside the registry:

```bash
PYTHONPATH=src python -m trackr_ml.cli evaluate --model-path /path/to/model.pkl
```

Important note: if you want a fully stable benchmark across model comparisons, freeze the files in `data/test/` between evaluations. In every case, they should contain both labels, represent production traffic, and never be reused for training or tuning.

## Prediction Output

Example:

```json
{
  "label": "financial_transaction",
  "confidence": 0.9471,
  "transaction": {
    "value": 55.9,
    "direction": "expense",
    "isCompleted": true,
    "is_pix": false,
    "card_type": "credit",
    "card_last4": "1234",
    "card_label": null
  }
}
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
