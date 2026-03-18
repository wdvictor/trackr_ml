from __future__ import annotations

import re
from typing import Literal

from .domain import TransactionDetails
from .text import compact_whitespace, normalize_text

AMOUNT_PATTERNS = (
    re.compile(r"r\$\s*(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})"),
    re.compile(r"(?<!\d)(\d{1,3}(?:\.\d{3})*,\d{2})(?!\d)"),
)

CARD_LAST4_PATTERNS = (
    re.compile(r"(?:final|terminad[oa]\s+em)\D{0,12}(\d{4})"),
    re.compile(r"\*{2,}\s*(\d{4})"),
    re.compile(r"x{2,}\s*(\d{4})"),
)

CARD_LABEL_PATTERNS = (
    re.compile(r"(?:cartão|card)\s+([a-z0-9 ]{2,40}?)(?:\s+final|\s+\*{2,}|\s+x{2,}|\s+\d{4})"),
    re.compile(r"(?:crédito|débito)\s+([a-z0-9 ]{2,40}?)(?:\s+final|\s+\*{2,}|\s+x{2,}|\s+\d{4})"),
)

PIX_KEYWORDS = (
    " pix ",
    "chave pix",
    "transferência pix",
    "transferência via pix",
    "pix enviado",
    "pix recebido",
)

INCOME_KEYWORDS = {
    "recebeu": 3,
    "recebido": 3,
    "recebemos": 2,
    "creditado": 3,
    "depósito": 2,
    "depósito recebido": 4,
    "estorno": 2,
    "reembolso": 2,
    "entrada": 1,
}

EXPENSE_KEYWORDS = {
    "pagamento": 3,
    "pagou": 3,
    "compra": 3,
    "débito": 2,
    "saque": 2,
    "transferência enviada": 4,
    "pix enviado": 4,
    "transacao aprovada": 3,
    "gasto": 2,
    "fatura": 2,
}

CREDIT_KEYWORDS = ("crédito", "cartão de crédito", "fatura", "parcelado")
DEBIT_KEYWORDS = ("débito", "cartao de débito", "funcao débito")


def extract_transaction_details(
    text: str, app_name: str | None = None
) -> TransactionDetails:
    normalized = f" {normalize_text(text)} "
    value = extract_amount(normalized)
    direction = detect_direction(normalized)
    card_type = detect_card_type(normalized)
    card_last4 = extract_card_last4(normalized)
    card_label = (
        extract_card_label(normalized, app_name)
        if card_type is not None or card_last4 is not None
        else None
    )

    return TransactionDetails(
        value=value,
        direction=direction,
        is_pix=contains_any(normalized, PIX_KEYWORDS),
        card_type=card_type,
        card_last4=card_last4,
        card_label=card_label,
    )


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_amount(text: str) -> float | None:
    for pattern in AMOUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        raw_amount = match.group(1).replace(".", "").replace(",", ".")
        try:
            return round(float(raw_amount), 2)
        except ValueError:
            continue

    return None


def detect_direction(text: str) -> Literal["income", "expense"] | None:
    income_score = keyword_score(text, INCOME_KEYWORDS)
    expense_score = keyword_score(text, EXPENSE_KEYWORDS)

    if income_score > expense_score:
        return "income"
    if expense_score > income_score:
        return "expense"
    return None


def keyword_score(text: str, keywords: dict[str, int]) -> int:
    score = 0
    for keyword, weight in keywords.items():
        if keyword in text:
            score += weight
    return score


def detect_card_type(text: str) -> Literal["credit", "debit"] | None:
    if any(keyword in text for keyword in CREDIT_KEYWORDS):
        return "credit"
    if any(keyword in text for keyword in DEBIT_KEYWORDS):
        return "debit"
    return None


def extract_card_last4(text: str) -> str | None:
    for pattern in CARD_LAST4_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def extract_card_label(text: str, app_name: str | None = None) -> str | None:
    ignored_labels = {"final", "cartao", "card", "credito", "debito"}

    for pattern in CARD_LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            label = compact_whitespace(match.group(1))
            if label and label not in ignored_labels:
                return label

    return None
