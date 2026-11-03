import os
import unittest
from unittest.mock import patch

from artefact_ai.config import Settings


class SettingsTests(unittest.TestCase):
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:secret@localhost/store"}, clear=True)
    def test_password_is_not_exposed_in_repr(self):
        settings = Settings.from_env()
        self.assertNotIn("secret", repr(settings))
        self.assertEqual(settings.history_turn_limit, 10)

    @patch.dict(os.environ, {"DATABASE_URL": "sqlite:///history.db"}, clear=True)
    def test_rejects_non_postgres_url(self):
        with self.assertRaises(ValueError):
            Settings.from_env()

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/store", "HISTORY_TURN_LIMIT": "0"}, clear=True)
    def test_rejects_empty_history_window(self):
        with self.assertRaises(ValueError):
            Settings.from_env()
