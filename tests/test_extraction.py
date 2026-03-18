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
        self.assertEqual(result.card_type, "credit")
        self.assertEqual(result.card_last4, "1234")
        self.assertEqual(result.card_label, "platinum")
        self.assertFalse(result.is_pix)

    def test_does_not_infer_card_without_card_signals(self) -> None:
        result = extract_transaction_details(
            text="Pix enviado de R$ 20,00 com sucesso.",
            app_name="com.picpay",
        )

        self.assertTrue(result.is_pix)
        self.assertIsNone(result.card_type)
        self.assertIsNone(result.card_last4)
        self.assertIsNone(result.card_label)


if __name__ == "__main__":
    unittest.main()
