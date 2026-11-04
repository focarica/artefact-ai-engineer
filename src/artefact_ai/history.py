"""Optional SQLite persistence for complete chat turns.

The default chat remains in memory. Supplying ``HISTORY_DB`` (or
``--history-db``) stores full PydanticAI turns, including tool messages, so a
session can be resumed in a later process without Docker or PostgreSQL.
"""

from pathlib import Path
import sqlite3
from typing import Protocol

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


class HistoryError(ValueError):
    """The local history file cannot be opened, read or written."""


class TurnStore(Protocol):
    def load(self, session_id: str, *, limit: int) -> list[list[ModelMessage]]: ...

    def append(self, session_id: str, messages: list[ModelMessage]) -> None: ...

    def prune(self, session_id: str, *, keep: int) -> None: ...


class SQLiteHistory:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path)
            self._connection.execute('PRAGMA foreign_keys = ON')
            self._connection.execute('PRAGMA journal_mode = WAL')
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    turn_number INTEGER NOT NULL CHECK (turn_number > 0),
                    messages TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, turn_number)
                );
                """
            )
        except (OSError, sqlite3.Error) as error:
            raise HistoryError(f'Não foi possível abrir o histórico: {self.path}') from error

    @staticmethod
    def _session(session_id: str) -> str:
        value = session_id.strip()
        if not value or len(value) > 128 or any(character in value for character in '\x00\n\r'):
            raise ValueError('session_id deve ter de 1 a 128 caracteres, sem quebras de linha.')
        return value

    @staticmethod
    def _messages(payload: str) -> list[ModelMessage]:
        try:
            messages = ModelMessagesTypeAdapter.validate_json(payload)
        except ValueError as error:
            raise HistoryError('O histórico salvo está corrompido e não pôde ser lido.') from error
        if not messages:
            raise HistoryError('O histórico salvo não contém mensagens.')
        return messages

    def load(self, session_id: str, *, limit: int) -> list[list[ModelMessage]]:
        if limit < 1:
            raise ValueError('limit must be positive')
        session = self._session(session_id)
        try:
            rows = self._connection.execute(
                'SELECT messages FROM turns WHERE session_id = ? ORDER BY turn_number DESC LIMIT ?',
                (session, limit),
            ).fetchall()
        except sqlite3.Error as error:
            raise HistoryError('Não foi possível ler o histórico salvo.') from error
        return [self._messages(payload) for (payload,) in reversed(rows)]

    def append(self, session_id: str, messages: list[ModelMessage]) -> None:
        session = self._session(session_id)
        payload = ModelMessagesTypeAdapter.dump_json(messages).decode('utf-8')
        try:
            with self._connection:
                self._connection.execute(
                    'INSERT OR IGNORE INTO sessions (session_id) VALUES (?)', (session,)
                )
                (turn_number,) = self._connection.execute(
                    'SELECT COALESCE(MAX(turn_number), 0) + 1 FROM turns WHERE session_id = ?',
                    (session,),
                ).fetchone()
                self._connection.execute(
                    'INSERT INTO turns (session_id, turn_number, messages) VALUES (?, ?, ?)',
                    (session, turn_number, payload),
                )
        except sqlite3.Error as error:
            raise HistoryError('Não foi possível gravar o histórico da sessão.') from error

    def prune(self, session_id: str, *, keep: int) -> None:
        if keep < 1:
            raise ValueError('keep must be positive')
        session = self._session(session_id)
        try:
            with self._connection:
                self._connection.execute(
                    'DELETE FROM turns WHERE session_id = ? AND turn_number <= '
                    '(SELECT MAX(turn_number) - ? FROM turns WHERE session_id = ?)',
                    (session, keep, session),
                )
        except sqlite3.Error as error:
            raise HistoryError('Não foi possível reduzir o histórico da sessão.') from error

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> 'SQLiteHistory':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
