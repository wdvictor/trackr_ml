from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PAGE_SIZE = 2000


def load_dotenv(env_path: Path | None = None) -> None:
    env_file = env_path or ROOT_DIR / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class Settings:
    api_url: str
    api_key: str
    raw_data_dir: Path
    test_data_dir: Path
    cache_dir: Path
    models_dir: Path
    page_size: int
    request_timeout_seconds: int
    unknown_lower_bound: float
    unknown_upper_bound: float

    @classmethod
    def from_env(cls, require_api: bool = True) -> "Settings":
        load_dotenv()

        api_url = os.getenv("NOTIFICATIONS_API_URL", "").strip()
        api_key = os.getenv("NOTIFICATIONS_API_KEY", "").strip()
        if require_api and (not api_url or not api_key):
            raise RuntimeError(
                "Configure NOTIFICATIONS_API_URL e NOTIFICATIONS_API_KEY no arquivo .env."
            )

        lower = float(os.getenv("UNKNOWN_LOWER_BOUND", "0.35"))
        upper = float(os.getenv("UNKNOWN_UPPER_BOUND", "0.65"))
        if lower >= upper:
            raise RuntimeError(
                "UNKNOWN_LOWER_BOUND deve ser menor que UNKNOWN_UPPER_BOUND."
            )

        return cls(
            api_url=api_url,
            api_key=api_key,
            raw_data_dir=ROOT_DIR / "data" / "raw",
            test_data_dir=ROOT_DIR / "data" / "test",
            cache_dir=ROOT_DIR / "data" / "cache",
            models_dir=ROOT_DIR / "models",
            page_size=DEFAULT_PAGE_SIZE,
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            unknown_lower_bound=lower,
            unknown_upper_bound=upper,
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_data_dir,
            self.test_data_dir,
            self.cache_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
