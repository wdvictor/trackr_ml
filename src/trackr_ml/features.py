from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from .text import normalize_text


class FieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field_name: str, normalizer: Any | None = None) -> None:
        self.field_name = field_name
        self.normalizer = normalizer

    def fit(self, X: list[dict[str, str]], y: list[int] | None = None) -> "FieldExtractor":
        return self

    def transform(self, X: list[dict[str, str]]) -> list[Any]:
        values = [item.get(self.field_name, "") for item in X]
        if self.normalizer is None:
            return values
        return [self.normalizer(value) for value in values]


class AppNameEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.is_fitted_ = False

    def fit(self, X: list[str], y: list[int] | None = None) -> "AppNameEncoder":
        self.encoder.fit([[value or "unknown"] for value in X])
        self.is_fitted_ = True
        return self

    def transform(self, X: list[str]):
        return self.encoder.transform([[value or "unknown"] for value in X])


def build_classifier_pipeline() -> Pipeline:
    feature_union = FeatureUnion(
        transformer_list=[
            (
                "word_text",
                Pipeline(
                    steps=[
                        ("selector", FieldExtractor("text", normalize_text)),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "char_text",
                Pipeline(
                    steps=[
                        ("selector", FieldExtractor("text", normalize_text)),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "app_name",
                Pipeline(
                    steps=[
                        ("selector", FieldExtractor("app_name", normalize_text)),
                        ("encoder", AppNameEncoder()),
                    ]
                ),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("features", feature_union),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
