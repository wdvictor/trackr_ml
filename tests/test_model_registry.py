from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trackr_ml.model_registry import (
    build_model_name,
    build_model_paths,
    list_registered_models,
    normalize_model_version,
    register_model,
    resolve_repo_path,
    resolve_registered_model,
)


class ModelRegistryTests(unittest.TestCase):
    def test_normalize_version_accepts_raw_or_prefixed_value(self) -> None:
        self.assertEqual(normalize_model_version("1.2.3"), "1.2.3")
        self.assertEqual(normalize_model_version("trackr-1.2.3"), "1.2.3")
        self.assertEqual(build_model_name("1.2.3"), "trackr-1.2.3")

    def test_build_paths_uses_trackr_prefix(self) -> None:
        paths = build_model_paths(Path("/tmp/models"), "2.0.0")

        self.assertEqual(paths["name"], "trackr-2.0.0")
        self.assertEqual(
            str(paths["model_path"]),
            "/tmp/models/trackr-2.0.0.pkl",
        )
        self.assertEqual(
            str(paths["metadata_path"]),
            "/tmp/models/trackr-2.0.0.json",
        )

    def test_register_and_resolve_latest_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            first = build_model_paths(models_dir, "1.0.0")
            second = build_model_paths(models_dir, "1.1.0")

            register_model(
                first["registry_path"],
                version="1.0.0",
                model_name=str(first["name"]),
                model_path=Path(first["model_path"]),
                metadata_path=Path(first["metadata_path"]),
                trained_at="2026-03-18T00:00:00+00:00",
                dataset_rows=10,
            )
            register_model(
                second["registry_path"],
                version="1.1.0",
                model_name=str(second["name"]),
                model_path=Path(second["model_path"]),
                metadata_path=Path(second["metadata_path"]),
                trained_at="2026-03-19T00:00:00+00:00",
                dataset_rows=12,
            )

            latest = resolve_registered_model(models_dir)
            explicit = resolve_registered_model(models_dir, "1.0.0")
            listed = list_registered_models(models_dir)

            self.assertEqual(latest["version"], "1.1.0")
            self.assertEqual(explicit["name"], "trackr-1.0.0")
            self.assertEqual([item["version"] for item in listed], ["1.1.0", "1.0.0"])
            self.assertFalse(Path(latest["model_path"]).is_absolute())
            self.assertFalse(Path(explicit["metadata_path"]).is_absolute())
            self.assertTrue(resolve_repo_path(latest["model_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
