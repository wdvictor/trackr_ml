# Como importar `trackr_ml` em outro projeto

## Forma escolhida

Este projeto nao sera publicado em PyPI ou registro privado. A forma adotada para reutiliza-lo em outro projeto Python e instalar o repositorio localmente em modo editavel.

Motivos:

- o pacote ja tem `pyproject.toml` e layout `src/`
- o import fica estavel com `from trackr_ml import run_predict`
- o codigo consumidor usa diretamente os artefatos e o `models/registry.json` mantidos neste repositorio
- alteracoes futuras neste repositorio passam a refletir no ambiente consumidor sem precisar publicar novas builds

## Como instalar no projeto consumidor

No ambiente virtual do outro projeto:

```bash
python -m pip install -e /absolute/path/to/trackr_ml
```

Exemplo:

```bash
python -m pip install -e /home/victor/Github/trackr_ml
```

Se preferir `uv`, o efeito pratico e o mesmo:

```bash
uv pip install -e /absolute/path/to/trackr_ml
```

## API publica suportada

A unica funcao exposta como API publica para reutilizacao e:

```python
from trackr_ml import run_predict
```

Assinatura:

```python
run_predict(
    text: str,
    app_name: str = "",
    model_path: str | Path | None = None,
    model_version: str | None = None,
) -> dict[str, object]
```

## Regras de selecao do modelo

- se `model_path` e `model_version` forem omitidos, usa o modelo mais recente registrado em `models/registry.json`
- se `model_version` for informado, usa a versao registrada correspondente
- se `model_path` for informado, usa exatamente esse arquivo `.pkl`
- se os dois forem informados ao mesmo tempo, a funcao gera erro

## Exemplo minimo

```python
from trackr_ml import run_predict

result = run_predict(
    text="Compra aprovada no credito final 1234 no valor de R$ 55,90",
    app_name="com.nu.production",
)

print(result)
```

Exemplo de retorno:

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

## Exemplo usando uma versao especifica

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    app_name="com.picpay",
    model_version="1.0.0",
)
```

## Exemplo usando caminho explicito

```python
from trackr_ml import run_predict

result = run_predict(
    text="Pix recebido de R$ 150,00",
    model_path="/absolute/path/to/model.pkl",
)
```

## Observacoes operacionais

- `run_predict` nao precisa de credenciais da API remota
- o projeto consumidor precisa ter acesso ao repositorio `trackr_ml` e aos artefatos em `models/`
- quando nenhum modelo e informado, o comportamento depende de existir um modelo registrado como mais recente em `models/registry.json`
