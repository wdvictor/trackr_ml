# How to import `trackr_ml` into another project

## Chosen approach

This project will not be published to PyPI or to a private package registry. The chosen way to reuse it in another Python project is to install the repository locally in editable mode.

Reasons:

- the package already has a `pyproject.toml` file and a `src/` layout
- the import path stays stable with `from trackr_ml import run_predict`
- the consumer code uses the artifacts and `models/registry.json` stored in this repository directly
- future changes in this repository become available to the consumer environment without publishing new builds

## How to install it in the consumer project

Inside the virtual environment of the other project:

```bash
python -m pip install -e /absolute/path/to/trackr_ml
```

Example:

```bash
python -m pip install -e /absolute/path/to/trackr_ml
```

If you prefer `uv`, the practical result is the same:

```bash
uv pip install -e /absolute/path/to/trackr_ml
```

## Supported public API

The only function exposed as a public API for reuse is:

```python
from trackr_ml import run_predict
```

Signature:

```python
run_predict(
    text: str,
    app_name: str = "",
    model_path: str | Path | None = None,
    model_version: str | None = None,
) -> dict[str, object]
```

## Model selection rules

- if `model_path` and `model_version` are both omitted, it uses the latest model registered in `models/registry.json`
- if `model_version` is provided, it uses the matching registered version
- if `model_path` is provided, it uses that exact `.pkl` file
- if both are provided at the same time, the function raises an error

## Minimal example

```python
from trackr_ml import run_predict

result = run_predict(
    text="Compra aprovada no credito final 1234 no valor de R$ 55,90",
    app_name="com.nu.production",
)

print(result)
```

Example return value:

```python
{
    "label": "financial_transaction",
    "confidence": 0.9471,
    "transaction": {
        "value": 55.9,
        "direction": "expense",
        "is_pix": False,
        "card_type": "credit",
        "card_last4": "1234",
        "card_label": None,
    },
}
```

## Example using a specific version

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    app_name="com.picpay",
    model_version="1.0.0",
)
```

## Example using an explicit path

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    model_path="/absolute/path/to/model.pkl",
)
```

## Operational notes

- `run_predict` does not require remote API credentials
- the consumer project must have access to the `trackr_ml` repository and the artifacts in `models/`
- when no model is provided, the behavior depends on having a latest registered model in `models/registry.json`
