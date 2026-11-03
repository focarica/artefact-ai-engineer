import json
import os
import unittest
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from artefact_ai.cli import app
from test_data import StoreFixture


class CliTests(StoreFixture, unittest.TestCase):
    def invoke(self, *arguments):
        return CliRunner().invoke(app, [*arguments, '--data-dir', str(self.directory)])

    def test_catalog_works_without_database_or_model(self):
        result = self.invoke('catalog', 'violao', '--max-price', '900')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)[0]['price_brl'], '899.91')

    def test_invalid_budget_is_a_usage_error(self):
        result = self.invoke('catalog', '--max-price', 'NaN')
        self.assertEqual(result.exit_code, 2)

    def test_wrong_customer_cannot_read_order(self):
        result = self.invoke('order', '1', '--customer-id', '2')
        self.assertEqual(result.exit_code, 1)
        self.assertNotIn('total_brl', result.output)

    def test_missing_data_has_actionable_error(self):
        result = CliRunner().invoke(app, ['catalog', '--data-dir', str(self.directory / 'missing')])
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Erro nos dados', result.output)

    @patch.dict(os.environ, {}, clear=True)
    def test_chat_single_message_works_without_database(self):
        def model(messages, info):
            self.assertIn('Empório da Música', info.instructions)
            return ModelResponse(parts=[TextPart('Oi! Como posso ajudar com seu instrumento?')])

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Empório: Oi!', result.output)

    def test_interactive_chat_keeps_context_and_skips_blank_input(self):
        prompts = []

        def model(messages, info):
            prompts.append([part.content for item in messages for part in item.parts
                            if isinstance(part, UserPromptPart)])
            return ModelResponse(parts=[TextPart('Posso ajudar!')])

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = CliRunner().invoke(app, ['chat', '--data-dir', str(self.directory)],
                                        input='\nOlá\nQuero um violão\n/sair\n')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(prompts, [['Olá'], ['Olá', 'Quero um violão']])

    def test_chat_eof_exits_cleanly(self):
        def model(messages, info):
            self.fail('EOF must not reach the model')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = CliRunner().invoke(app, ['chat', '--data-dir', str(self.directory)], input='')
        self.assertEqual(result.exit_code, 0, result.output)

    def test_empty_chat_message_is_usage_error(self):
        with patch('artefact_ai.agent.create_model') as model:
            result = self.invoke('chat', '--message', '   ')
        self.assertEqual(result.exit_code, 2, result.output)
        model.assert_not_called()

    def test_chat_auth_error_explains_container_signin(self):
        def model(messages, info):
            raise ModelHTTPError(status_code=401, model_name='test', body='private provider details')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('docker compose exec ollama ollama signin', result.output)
        self.assertNotIn('private provider details', result.output)

    def test_chat_timeout_is_actionable_without_traceback(self):
        def model(messages, info):
            raise TimeoutError('private details')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Verifique o Ollama', result.output)
        self.assertNotIn('private details', result.output)

    @patch.dict(os.environ, {'OLLAMA_BASE_URL': 'invalid'}, clear=True)
    def test_chat_invalid_configuration_is_actionable(self):
        result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Configuração inválida', result.output)
