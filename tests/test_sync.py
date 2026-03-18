from __future__ import annotations

import unittest

from trackr_ml.sync import sanitize_api_url


class SyncTests(unittest.TestCase):
    def test_sanitize_api_url_removes_managed_params(self) -> None:
        url = (
            "https://notifications.example/api/notifications"
            "?p=1&size=2000&isft=true&foo=bar"
        )

        self.assertEqual(
            sanitize_api_url(url),
            "https://notifications.example/api/notifications?foo=bar",
        )


if __name__ == "__main__":
    unittest.main()
