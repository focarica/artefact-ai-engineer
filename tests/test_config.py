import os
from pathlib import Path
import unittest
from unittest.mock import patch

from artefact_ai.config import ChatSettings


class ChatSettingsTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_do_not_require_database_or_api_key(self):
        settings = ChatSettings.from_env()
        self.assertEqual(settings.ollama_model, 'gpt-oss:120b')
        self.assertEqual(settings.ollama_base_url, 'https://ollama.com/v1')
        self.assertEqual(settings.history_turn_limit, 10)
        self.assertIsNone(settings.history_db)

    @patch.dict(os.environ, {'OLLAMA_MODEL': 'gpt-oss:120b:cloud'}, clear=True)
    def test_cloud_alias_is_normalized_for_direct_api(self):
        self.assertEqual(ChatSettings.from_env().ollama_model, 'gpt-oss:120b')

    @patch.dict(os.environ, {'OLLAMA_API_KEY': 'private-api-key'}, clear=True)
    def test_api_key_is_not_exposed_in_repr(self):
        settings = ChatSettings.from_env()
        self.assertEqual(settings.ollama_api_key, 'private-api-key')
        self.assertNotIn('private-api-key', repr(settings))

    @patch.dict(os.environ, {'HISTORY_DB': '~/artefact-history.sqlite3'}, clear=True)
    def test_history_database_path_is_expanded(self):
        expected = Path('~/artefact-history.sqlite3').expanduser().resolve()
        self.assertEqual(ChatSettings.from_env().history_db, expected)

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
