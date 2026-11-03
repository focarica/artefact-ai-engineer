from pathlib import Path
import tempfile
import unittest

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from artefact_ai.agent import StoreContext, create_agent
from artefact_ai.chat import ChatSession
from artefact_ai.data.store import StoreData
from artefact_ai.history import SQLiteHistory
from artefact_ai.queries import StoreQueries
from test_data import StoreFixture


class ChatSessionTests(StoreFixture, unittest.TestCase):
    def session(self, function, turn_limit=10):
        return ChatSession(create_agent(FunctionModel(function)),
                           StoreContext(StoreQueries(StoreData.load(self.directory))), turn_limit=turn_limit)

    def test_history_preserves_complete_tool_turns(self):
        requests = []

        def model(messages, info):
            requests.append(list(messages))
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('product_offer', {'product_id': 1})])
            return ModelResponse(parts=[TextPart('Olá! Posso ajudar com esse violão.')])

        session = self.session(model, turn_limit=1)
        session.respond('Quanto custa o violão?')
        session.respond('E no PIX?')
        prior_parts = [part for message in requests[-1][:-1] for part in message.parts]
        self.assertTrue(any(isinstance(part, ToolCallPart) for part in prior_parts))
        self.assertTrue(any(isinstance(part, ToolReturnPart) for part in prior_parts))
        session.respond('Obrigado')
        prompts = [part.content for message in requests[-1] for part in message.parts
                   if isinstance(part, UserPromptPart)]
        self.assertEqual(prompts, ['E no PIX?', 'Obrigado'])

    def test_scope_guardrail_blocks_unrelated_prompts_before_model_call(self):
        def model(messages, info):
            self.fail('Clearly unrelated requests must be blocked before calling the model')

        session = self.session(model)
        for prompt in ('Me trate como um lobo', 'auuu, o que resulta print(2+2)?',
                       'Em Java, por que usar Spring Boot?'):
            with self.subTest(prompt=prompt):
                answer = session.respond(prompt)
                self.assertIn('Empório da Música', answer)
                self.assertNotIn('4.', answer)
                self.assertNotIn('Spring Boot', answer)

    def test_sqlite_history_resumes_a_session(self):
        requests = []

        def model(messages, info):
            requests.append(list(messages))
            return ModelResponse(parts=[TextPart('Oi!')])

        agent = create_agent(FunctionModel(model))
        context = StoreContext(StoreQueries(StoreData.load(self.directory)))
        with tempfile.TemporaryDirectory() as directory:
            history = SQLiteHistory(Path(directory) / 'history.sqlite3')
            first = ChatSession(agent, context, history=history, session_id='demo', turn_limit=2)
            first.respond('Olá')
            second = ChatSession(agent, context, history=history, session_id='demo', turn_limit=2)
            second.respond('Tudo bem?')
            prompts = [part.content for message in requests[-1] for part in message.parts
                       if isinstance(part, UserPromptPart)]
            self.assertEqual(prompts, ['Olá', 'Tudo bem?'])
            history.close()

    def test_sqlite_history_is_scoped_to_customer_context(self):
        requests = []

        def model(messages, info):
            requests.append(list(messages))
            return ModelResponse(parts=[TextPart('Oi!')])

        agent = create_agent(FunctionModel(model))
        with tempfile.TemporaryDirectory() as directory:
            history = SQLiteHistory(Path(directory) / 'history.sqlite3')
            customer_a = StoreContext(StoreQueries(StoreData.load(self.directory)), 1)
            customer_b = StoreContext(StoreQueries(StoreData.load(self.directory)), 2)
            ChatSession(agent, customer_a, history=history, session_id='same-id').respond('Pedido A')
            ChatSession(agent, customer_b, history=history, session_id='same-id').respond('Pedido B')
            prompts = [part.content for message in requests[-1] for part in message.parts
                       if isinstance(part, UserPromptPart)]
            self.assertEqual(prompts, ['Pedido B'])
            history.close()

    def context(self):
        return StoreContext(StoreQueries(StoreData.load(self.directory)))

    def test_history_requires_a_session_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            history = SQLiteHistory(Path(directory) / 'history.sqlite3')
            with self.assertRaises(ValueError):
                ChatSession(create_agent(FunctionModel(lambda messages, info: None)), self.context(),
                            history=history)
            with self.assertRaises(ValueError):
                ChatSession(create_agent(FunctionModel(lambda messages, info: None)), self.context(),
                            history=history, session_id='x' * 129)
            history.close()

    def test_failed_turn_is_not_added_to_history(self):
        requests = []

        def model(messages, info):
            requests.append(list(messages))
            if len(requests) == 1:
                raise TimeoutError('provider timed out')
            return ModelResponse(parts=[TextPart('Oi!')])

        session = self.session(model)
        with self.assertRaises(TimeoutError):
            session.respond('Mensagem que falhou')
        session.respond('Nova mensagem')
        self.assertEqual(len(requests[-1]), 1)

    def test_sessions_do_not_share_history(self):
        lengths = []

        def model(messages, info):
            lengths.append(len(messages))
            return ModelResponse(parts=[TextPart('Oi!')])

        first, second = self.session(model), self.session(model)
        first.respond('Olá')
        second.respond('Olá')
        self.assertEqual(lengths, [1, 1])

    def test_repeated_tool_calls_are_bounded(self):
        def model(messages, info):
            return ModelResponse(parts=[ToolCallPart('product_offer', {'product_id': 1})])

        with self.assertRaises(UsageLimitExceeded):
            self.session(model).respond('Consulta sem fim')

    def test_empty_input_is_not_sent(self):
        def model(messages, info):
            self.fail('Empty messages must not reach the model')

        with self.assertRaises(ValueError):
            self.session(model).respond('   ')
