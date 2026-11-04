import sqlite3
import tempfile
from pathlib import Path
import unittest

from pydantic_ai.messages import ModelResponse, TextPart

from artefact_ai.history import HistoryError, SQLiteHistory


class SQLiteHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'history.sqlite3'
        self.history = SQLiteHistory(self.path)
        self.addCleanup(self.history.close)

    def turn(self, text):
        return [ModelResponse(parts=[TextPart(text)])]

    def test_sessions_are_isolated(self):
        self.history.append('cliente-a', self.turn('a'))
        self.history.append('cliente-b', self.turn('b'))
        self.assertEqual(self.history.load('cliente-a', limit=10)[0][0].parts[0].content, 'a')
        self.assertEqual(self.history.load('cliente-b', limit=10)[0][0].parts[0].content, 'b')

    def test_prune_removes_old_turns_but_keeps_the_latest(self):
        for number in range(1, 4):
            self.history.append('cliente-a', self.turn(str(number)))
        self.history.prune('cliente-a', keep=2)
        loaded = self.history.load('cliente-a', limit=10)
        self.assertEqual([turn[0].parts[0].content for turn in loaded], ['2', '3'])
        newest = self.history.load('cliente-a', limit=1)
        self.assertEqual(newest[0][0].parts[0].content, '3')

    def test_corrupted_history_is_actionable(self):
        self.history.append('cliente-a', self.turn('ok'))
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE turns SET messages = 'not json' WHERE session_id = 'cliente-a'")
        with self.assertRaisesRegex(HistoryError, 'corrompido'):
            self.history.load('cliente-a', limit=10)

    def test_invalid_session_identifier_is_rejected(self):
        for session_id in ['', '   ', 'a' * 129, 'linha\nquebrada']:
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                self.history.load(session_id, limit=1)
