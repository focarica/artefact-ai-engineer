"""Application configuration read from the process environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def history_turn_limit() -> int:
    limit = int(os.environ.get("HISTORY_TURN_LIMIT", "10"))
    if limit < 1:
        raise ValueError("HISTORY_TURN_LIMIT must be positive")
    return limit


@dataclass(frozen=True)
class ChatSettings:
    """Chat uses Ollama Cloud and CSVs; SQLite history is optional."""

    ollama_base_url: str
    ollama_model: str
    data_dir: Path
    history_turn_limit: int = 10
    ollama_api_key: str = field(default='', repr=False)
    history_db: Path | None = None

    @classmethod
    def from_env(cls) -> "ChatSettings":
        base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1").strip().rstrip("/")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL ending in /v1")
        if parsed.path != "/v1" or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("OLLAMA_BASE_URL must end in /v1 without credentials, query or fragment")
        # Accessing port also validates malformed or out-of-range ports.
        parsed.port
        model = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL must not be empty")
        if parsed.hostname == 'ollama.com':
            model = model.removesuffix(':cloud')
        history_db = os.environ.get("HISTORY_DB", "").strip()
        return cls(
            base_url,
            model,
            Path(os.environ.get("DATA_DIR", "../data")).expanduser().resolve(),
            history_turn_limit(),
            os.environ.get('OLLAMA_API_KEY', '').strip(),
            Path(history_db).expanduser().resolve() if history_db else None,
        )
