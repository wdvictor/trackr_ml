from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import trackr_ml
from trackr_ml.api import run_predict


class PublicApiTests(unittest.TestCase):
    def test_package_exports_run_predict(self) -> None:
        self.assertIs(trackr_ml.run_predict, run_predict)
        self.assertEqual(trackr_ml.__all__, ["run_predict"])

    @patch("trackr_ml.api.NotificationClassifier.load")
    @patch("trackr_ml.api.Settings.from_env")
    def test_run_predict_uses_latest_registered_model_when_model_is_omitted(
        self,
        from_env_mock: Mock,
        load_mock: Mock,
    ) -> None:
        classifier = Mock()
        prediction = Mock()
        prediction.to_dict.return_value = {
            "label": "unknown",
            "confidence": 0.0,
            "transaction": None,
        }
        classifier.predict.return_value = prediction
        load_mock.return_value = classifier

        payload = run_predict(text="Teste")

        from_env_mock.assert_called_once_with(require_api=False)
        load_mock.assert_called_once_with(model_path=None, model_version=None)
        classifier.predict.assert_called_once_with(text="Teste", app_name="")
        self.assertEqual(
            payload,
            {
                "label": "unknown",
                "confidence": 0.0,
                "transaction": None,
            },
        )

    @patch("trackr_ml.api.NotificationClassifier.load")
    @patch("trackr_ml.api.Settings.from_env")
    def test_run_predict_accepts_string_model_path(
        self,
        from_env_mock: Mock,
        load_mock: Mock,
    ) -> None:
        classifier = Mock()
        prediction = Mock()
        prediction.to_dict.return_value = {"label": "financial_transaction"}
        classifier.predict.return_value = prediction
        load_mock.return_value = classifier

        run_predict(
            text="Compra aprovada",
            app_name="com.nu.production",
            model_path="/tmp/model.pkl",
        )

        from_env_mock.assert_called_once_with(require_api=False)
        load_mock.assert_called_once_with(
            model_path=Path("/tmp/model.pkl"),
            model_version=None,
        )
        classifier.predict.assert_called_once_with(
            text="Compra aprovada",
            app_name="com.nu.production",
        )

    def test_run_predict_rejects_model_path_and_version_together(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Use apenas model_path ou model_version, nunca os dois.",
        ):
            run_predict(
                text="Compra aprovada",
                model_path="/tmp/model.pkl",
                model_version="1.0.0",
            )


if __name__ == "__main__":
    unittest.main()
