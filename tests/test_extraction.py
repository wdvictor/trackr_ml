from __future__ import annotations

import unittest

from trackr_ml.extraction import extract_transaction_details


class ExtractionTests(unittest.TestCase):
    def test_extracts_pix_income_amount(self) -> None:
        result = extract_transaction_details(
            text="Voce recebeu um pix de R$ 1.250,89 na sua conta.",
            app_name="com.nu.production",
        )

        self.assertEqual(result.value, 1250.89)
        self.assertEqual(result.direction, "income")
        self.assertTrue(result.is_completed)
        self.assertTrue(result.is_pix)
        self.assertIsNone(result.card_type)
        self.assertIsNone(result.card_last4)
        self.assertIsNone(result.card_label)

    def test_extracts_credit_card_metadata(self) -> None:
        result = extract_transaction_details(
            text="Compra aprovada no credito cartao platinum final 1234 no valor de R$ 55,90.",
            app_name="com.itau",
        )

        self.assertEqual(result.value, 55.90)
        self.assertEqual(result.direction, "expense")
        self.assertTrue(result.is_completed)
        self.assertEqual(result.card_type, "credit")
        self.assertEqual(result.card_last4, "1234")
        self.assertEqual(result.card_label, "platinum")
        self.assertFalse(result.is_pix)

    def test_extracts_declined_transaction_status(self) -> None:
        result = extract_transaction_details(
            text=(
                "compra recusada: limite insuficiente a compra no cartao final "
                "5371********7761 em 19/03/2026, de r$ 26.90, em google "
                "youtubepremium sao paulo bra, foi recusada por limite de "
                "credito insuficiente. toque para conferir seu limite"
            ),
            app_name="com.c6bank.app",
        )

        self.assertEqual(result.value, 26.90)
        self.assertEqual(result.direction, "expense")
        self.assertFalse(result.is_completed)
        self.assertEqual(result.card_type, "credit")
        self.assertEqual(result.card_last4, "7761")
        self.assertFalse(result.is_pix)

    def test_returns_unknow_when_completion_status_is_not_clear(self) -> None:
        result = extract_transaction_details(
            text="Compra em processamento no cartao final 1234 no valor de R$ 55,90.",
            app_name="com.itau",
        )

        self.assertEqual(result.value, 55.90)
        self.assertEqual(result.direction, "expense")
        self.assertEqual(result.is_completed, "unknow")
        self.assertEqual(result.card_last4, "1234")

    def test_does_not_infer_card_without_card_signals(self) -> None:
        result = extract_transaction_details(
            text="Pix enviado de R$ 20,00 com sucesso.",
            app_name="com.picpay",
        )

        self.assertTrue(result.is_completed)
        self.assertTrue(result.is_pix)
        self.assertIsNone(result.card_type)
        self.assertIsNone(result.card_last4)
        self.assertIsNone(result.card_label)


if __name__ == "__main__":
    unittest.main()
