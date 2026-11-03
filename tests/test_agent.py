from decimal import Decimal
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from artefact_ai.agent import PERSONA, StoreContext, create_agent, create_model
from artefact_ai.config import ChatSettings
from artefact_ai.data.store import StoreData
from artefact_ai.queries import StoreQueries
from pydantic_ai.providers.ollama import OllamaProvider
from test_data import StoreFixture


class AgentTests(StoreFixture, unittest.TestCase):
    def context(self, customer_id=None):
        return StoreContext(StoreQueries(StoreData.load(self.directory)), customer_id)

    def call_tool(self, name, arguments, customer_id=None):
        returned = []

        def model(messages, info):
            self.assertEqual(info.instructions, PERSONA.strip())
            order_tool = next(tool for tool in info.function_tools if tool.name == 'order_summary')
            self.assertEqual(set(order_tool.parameters_json_schema['properties']), {'order_id'})
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(name, arguments)])
            returned.extend(part.content for part in messages[-1].parts if isinstance(part, ToolReturnPart))
            return ModelResponse(parts=[TextPart('Consulta concluída.')])

        agent = create_agent(FunctionModel(model))
        result = agent.run_sync('Faça a consulta.', deps=self.context(customer_id))
        self.assertEqual(result.output, 'Consulta concluída.')
        self.assertEqual(len(returned), 1)
        return returned[0]

    def test_catalog_tool_preserves_pricing_and_budget(self):
        offers = self.call_tool('search_products', {'query': 'violao', 'max_price_brl': '900'})
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].price_brl, Decimal('899.91'))
        self.assertEqual(offers[0].pix_price_brl, Decimal('899.91'))

    def test_product_tool_returns_offer(self):
        offer = self.call_tool('product_offer', {'product_id': 1})
        self.assertEqual(offer.product_id, 1)

    def test_order_identity_comes_from_session(self):
        order = self.call_tool('order_summary', {'order_id': 1}, customer_id=1)
        self.assertEqual(order.order_id, 1)
        self.assertFalse(hasattr(order, 'customer_id'))

    def test_wrong_customer_and_missing_order_are_indistinguishable(self):
        self.assertIsNone(self.call_tool('order_summary', {'order_id': 1}, customer_id=2))
        self.assertIsNone(self.call_tool('order_summary', {'order_id': 999}, customer_id=2))

    def test_orders_without_session_identity_are_unavailable(self):
        self.assertIn('Não há cliente', self.call_tool('order_summary', {'order_id': 1}))

    def test_cloud_model_requires_api_key_before_connecting(self):
        settings = ChatSettings('https://ollama.com/v1', 'deepseek-v4.1-flash', self.directory)
        with self.assertRaisesRegex(ValueError, 'OLLAMA_API_KEY'):
            create_model(settings)

    def test_cloud_request_sends_api_key_model_and_persona(self):
        requests = []

        def handle(request):
            self.assertEqual(str(request.url), 'https://ollama.com/v1/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer test-key')
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body['model'], 'deepseek-v4.1-flash')
            self.assertTrue(any('Empório da Música' in message.get('content', '')
                                for message in body['messages'] if isinstance(message.get('content'), str)))
            return httpx.Response(200, json={
                'id': 'test-response', 'object': 'chat.completion', 'created': 0,
                'model': body['model'], 'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'Oi! Como posso ajudar?'},
                     'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18},
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        provider = OllamaProvider(base_url='https://ollama.com/v1', api_key='test-key', http_client=client)
        self.addCleanup(lambda: asyncio.run(provider.client.close()))
        settings = ChatSettings('https://ollama.com/v1', 'deepseek-v4.1-flash', self.directory,
                                ollama_api_key='test-key')
        with patch('artefact_ai.agent.OllamaProvider', return_value=provider) as factory:
            agent = create_agent(create_model(settings))
        factory.assert_called_once_with(base_url='https://ollama.com/v1', api_key='test-key')
        self.assertEqual(agent.run_sync('Olá', deps=self.context()).output, 'Oi! Como posso ajudar?')
        self.assertEqual(len(requests), 1)
