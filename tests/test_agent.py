from decimal import Decimal
import unittest

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from artefact_ai.agent import PERSONA, StoreContext, create_agent
from artefact_ai.data.store import StoreData
from artefact_ai.queries import StoreQueries
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
