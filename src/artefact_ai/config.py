"""Application configuration read from the process environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    data_dir: Path
    history_turn_limit: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        database_url = os.environ.get("DATABASE_URL", "")
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
        if not parsed.path.strip("/"):
            raise ValueError("DATABASE_URL must include a database name")
        limit = int(os.environ.get("HISTORY_TURN_LIMIT", "10"))
        if limit < 1:
            raise ValueError("HISTORY_TURN_LIMIT must be positive")
        data_dir = Path(os.environ.get("DATA_DIR", "../data")).expanduser().resolve()
        return cls(database_url=database_url, data_dir=data_dir, history_turn_limit=limit)
