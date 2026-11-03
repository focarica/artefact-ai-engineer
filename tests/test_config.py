import os
import unittest
from unittest.mock import patch

from artefact_ai.config import ChatSettings, Settings


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


class ChatSettingsTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_do_not_require_database_or_api_key(self):
        settings = ChatSettings.from_env()
        self.assertEqual(settings.ollama_model, 'deepseek-v4.1-flash:cloud')
        self.assertEqual(settings.ollama_base_url, 'http://localhost:11434/v1')
        self.assertEqual(settings.history_turn_limit, 10)

    @patch.dict(os.environ, {'OLLAMA_PORT': '11435'}, clear=True)
    def test_port_override_matches_compose(self):
        self.assertEqual(ChatSettings.from_env().ollama_base_url, 'http://localhost:11435/v1')

    @patch.dict(os.environ, {'OLLAMA_BASE_URL': 'http://ollama:11434/v1/'}, clear=True)
    def test_explicit_url_takes_precedence(self):
        self.assertEqual(ChatSettings.from_env().ollama_base_url, 'http://ollama:11434/v1')

    def test_rejects_invalid_configuration(self):
        for environment in [
            {'OLLAMA_BASE_URL': 'file:///v1'},
            {'OLLAMA_BASE_URL': 'http://localhost:11434'},
            {'OLLAMA_BASE_URL': 'http://user:secret@localhost/v1'},
            {'OLLAMA_BASE_URL': 'http://localhost:99999/v1'},
            {'OLLAMA_MODEL': ' '},
            {'HISTORY_TURN_LIMIT': '0'},
        ]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    ChatSettings.from_env()
