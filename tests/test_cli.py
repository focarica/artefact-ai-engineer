import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from artefact_ai.cli import app
from test_data import StoreFixture
from test_manual import write_manual_pdf


class CliTests(StoreFixture, unittest.TestCase):
    def invoke(self, *arguments):
        arguments = list(arguments)
        if arguments and arguments[0] == 'chat' and '--history-db' not in arguments and '--no-history' not in arguments:
            arguments.append('--no-history')
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

    def test_policies_command_does_not_require_csvs_or_model(self):
        path = self.directory / 'políticas_da_loja.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        result = CliRunner().invoke(app, ['policies', 'horário', '--manual-path', str(path),
                                        '--data-dir', str(self.directory / 'missing-csvs')])
        self.assertEqual(result.exit_code, 0, result.output)
        passage = json.loads(result.output)[0]
        self.assertEqual(passage['page'], 1)
        self.assertIn('09:00', passage['text'])

    def test_policies_missing_pdf_is_actionable(self):
        result = self.invoke('policies', 'frete')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Erro no manual', result.output)

    def test_chat_uses_policy_pdf_from_selected_data_directory(self):
        write_manual_pdf(self.directory / 'políticas_da_loja.pdf',
                         ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        returns = []

        def model(messages, info):
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário sábado'})])
            returns.extend(part.content for part in messages[-1].parts if isinstance(part, ToolReturnPart))
            return ModelResponse(parts=[TextPart('Sábado: 09:00 às 13:00 (manual, seção 2, p. 1).')])

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Vocês abrem no sábado?')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(returns[0].status, 'ok')
        self.assertIn('p. 1', result.output)

    def test_chat_policy_path_can_be_overridden(self):
        custom = self.directory / 'custom.pdf'
        write_manual_pdf(custom, ['2. Horario de Funcionamento', 'Sabado 10:00 as 12:00'])

        def model(messages, info):
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
            returned = next(p.content for p in messages[-1].parts if isinstance(p, ToolReturnPart))
            self.assertEqual(returned.passages[0].source, 'custom.pdf')
            return ModelResponse(parts=[TextPart('Sábado: 10:00 às 12:00 (manual, seção 2, página 1).')])

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--manual-path', str(custom), '--message', 'Qual o horário?')
        self.assertEqual(result.exit_code, 0, result.output)

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
            result = CliRunner().invoke(app, ['chat', '--no-history', '--data-dir', str(self.directory)],
                                        input='\nOlá\nQuero um violão\n/sair\n')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(prompts, [['Olá'], ['Olá', 'Quero um violão']])

    def test_configured_history_database_is_used_by_default(self):
        def model(messages, info):
            return ModelResponse(parts=[TextPart('Oi!')])

        database = self.directory / 'default-history.sqlite3'
        with patch.dict(os.environ, {'HISTORY_DB': str(database)}), \
             patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = CliRunner().invoke(app, ['chat', '--data-dir', str(self.directory), '--message', 'Olá'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(database.exists())
        self.assertIn('Sessão salva. Para retomá-la', result.output)
        self.assertIn('--history-db', result.output)
        self.assertRegex(result.output, r'--session-id [0-9a-f]{32}')

    def test_default_history_file_is_omitted_from_resume_command(self):
        def model(messages, info):
            return ModelResponse(parts=[TextPart('Oi!')])

        with patch.dict(os.environ, {}, clear=True), \
             patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            working_directory = self.directory / 'default-db-cwd'
            working_directory.mkdir()
            previous_directory = Path.cwd()
            os.chdir(working_directory)
            try:
                result = CliRunner().invoke(
                    app, ['chat', '--data-dir', str(self.directory), '--message', 'Olá'])
                self.assertTrue(Path('artefact_history.sqlite3').exists())
            finally:
                os.chdir(previous_directory)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('--session-id', result.output)
        self.assertNotIn('--history-db', result.output)

    def test_session_id_is_not_shown_until_chat_exits(self):
        def model(messages, info):
            return ModelResponse(parts=[TextPart('Estou aqui para ajudar.')])

        database = self.directory / 'interactive-history.sqlite3'
        with patch.dict(os.environ, {'HISTORY_DB': str(database)}), \
             patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = CliRunner().invoke(app, ['chat', '--data-dir', str(self.directory)],
                                        input='Olá\n/sair\n')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn('Sessão persistente:', result.output)
        self.assertIn('Histórico salvo em SQLite', result.output)
        self.assertIn('Sessão salva. Para retomá-la', result.output)

    def test_chat_persists_and_resumes_a_sqlite_session(self):
        histories = []

        def model(messages, info):
            histories.append([part.content for message in messages for part in message.parts
                              if isinstance(part, UserPromptPart)])
            return ModelResponse(parts=[TextPart('Oi! Como posso ajudar?')])

        database = self.directory / 'history.sqlite3'
        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            first = self.invoke('chat', '--history-db', str(database), '--session-id', 'demo',
                                '--message', 'Olá')
            second = self.invoke('chat', '--history-db', str(database), '--session-id', 'demo',
                                 '--message', 'Tudo bem?')
        self.assertEqual(first.exit_code, 0, first.output)
        self.assertEqual(second.exit_code, 0, second.output)
        self.assertTrue(database.exists())
        self.assertEqual(histories[0], ['Olá'])
        self.assertEqual(histories[-1], ['Olá', 'Tudo bem?'])

    def test_empty_chat_message_is_usage_error(self):
        with patch('artefact_ai.agent.create_model') as model:
            result = self.invoke('chat', '--message', '   ')
        self.assertEqual(result.exit_code, 2, result.output)
        model.assert_not_called()

    def test_chat_auth_error_explains_api_key(self):
        def model(messages, info):
            raise ModelHTTPError(status_code=401, model_name='test', body='private provider details')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Verifique OLLAMA_API_KEY', result.output)
        self.assertNotIn('private provider details', result.output)

    def test_chat_timeout_is_actionable_without_traceback(self):
        def model(messages, info):
            raise TimeoutError('private details')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Verifique o Ollama', result.output)
        self.assertNotIn('private details', result.output)

    def test_unreliable_answer_is_not_reported_as_connection_error(self):
        def model(messages, info):
            raise UnexpectedModelBehavior('private details')

        with patch('artefact_ai.agent.create_model', return_value=FunctionModel(model)):
            result = self.invoke('chat', '--message', 'Qual o frete?')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('consistente com as fontes', result.output)
        self.assertNotIn('sua conexão', result.output)
        self.assertNotIn('private details', result.output)

    @patch.dict(os.environ, {'OLLAMA_BASE_URL': 'invalid'}, clear=True)
    def test_chat_invalid_configuration_is_actionable(self):
        result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Configuração inválida', result.output)

    @patch.dict(os.environ, {}, clear=True)
    def test_chat_missing_api_key_is_actionable(self):
        result = self.invoke('chat', '--message', 'Olá')
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn('Defina OLLAMA_API_KEY', result.output)


class EntrypointTests(unittest.TestCase):
    @patch.dict(os.environ, {'OLLAMA_MODEL': 'environment-model'}, clear=True)
    def test_main_loads_local_dotenv_without_overriding_environment(self):
        from artefact_ai.cli import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / '.env').write_text('OLLAMA_API_KEY=test-key\nOLLAMA_MODEL=file-model\n')

            def check_environment():
                self.assertEqual(os.environ['OLLAMA_API_KEY'], 'test-key')
                self.assertEqual(os.environ['OLLAMA_MODEL'], 'environment-model')

            with patch('artefact_ai.cli.Path.cwd', return_value=path), \
                    patch('artefact_ai.cli.app', side_effect=check_environment) as application:
                main()
            application.assert_called_once_with()
