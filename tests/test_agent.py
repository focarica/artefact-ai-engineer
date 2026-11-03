from decimal import Decimal
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from artefact_ai.agent import PERSONA, StoreContext, clock_times, create_agent, create_model, phone_numbers
from artefact_ai.config import ChatSettings
from artefact_ai.data.store import StoreData
from artefact_ai.manual import PolicySearchResult
from artefact_ai.queries import StoreQueries
from pydantic_ai.providers.ollama import OllamaProvider
from test_data import StoreFixture
from test_manual import write_manual_pdf


class AgentTests(StoreFixture, unittest.TestCase):
    def context(self, customer_id=None, manual_path=None):
        return StoreContext(StoreQueries(StoreData.load(self.directory)), customer_id, manual_path)

    def call_tool(self, name, arguments, customer_id=None, manual_path=None):
        returned = []

        def model(messages, info):
            self.assertEqual(info.instructions, PERSONA.strip())
            order_tool = next(tool for tool in info.function_tools if tool.name == 'order_summary')
            self.assertEqual(set(order_tool.parameters_json_schema['properties']), {'order_id'})
            policy_tool = next(tool for tool in info.function_tools if tool.name == 'search_policies')
            self.assertEqual(set(policy_tool.parameters_json_schema['properties']), {'query'})
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(name, arguments)])
            returned.extend(part.content for part in messages[-1].parts if isinstance(part, ToolReturnPart))
            answer = 'Consulta concluída.'
            if returned and isinstance(returned[-1], PolicySearchResult):
                if returned[-1].passages:
                    passage = returned[-1].passages[0]
                    answer += f' (Manual, seção {passage.section}, página {passage.page}.)'
                else:
                    answer += ' Não consegui confirmar; confirme com a equipe.'
            return ModelResponse(parts=[TextPart(answer)])

        agent = create_agent(FunctionModel(model))
        result = agent.run_sync('Faça a consulta.', deps=self.context(customer_id, manual_path))
        self.assertTrue(result.output.startswith('Consulta concluída.'))
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

    def test_policy_tool_returns_source_page_and_text(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        result = self.call_tool('search_policies', {'query': 'horário sábado'}, manual_path=path)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.passages[0].page, 1)
        self.assertEqual(result.passages[0].source, 'manual.pdf')
        self.assertIn('09:00', result.passages[0].text)

    def test_policy_tool_distinguishes_unrelated_query_from_missing_manual(self):
        path = self.directory / 'manual.pdf'
        self.assertEqual(self.call_tool('search_policies', {'query': 'frete'}).status, 'unavailable')
        self.assertEqual(self.call_tool('search_policies', {'query': 'frete'}, manual_path=path).status, 'unavailable')
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        result = self.call_tool('search_policies', {'query': 'fotossintese'}, manual_path=path)
        self.assertEqual(result.status, 'not_found')
        self.assertEqual(result.passages, ())

    def test_explicit_source_request_requires_reference(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
            answer = 'Abre às 09:00.' if len(calls) == 2 else 'Abre às 09:00 (manual, seção 2, página 1).'
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Qual o horário? Cite a fonte.', deps=self.context(manual_path=path))
        self.assertIn('página 1', result.output)
        self.assertEqual(len(calls), 3)

    def test_normal_policy_answer_does_not_require_visible_citations(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
            return ModelResponse(parts=[TextPart('Na Empório, abrimos aos sábados das 9h às 13h!')])

        result = create_agent(FunctionModel(model)).run_sync('Qual o horário de sábado?',
                                                           deps=self.context(manual_path=path))
        self.assertEqual(result.output, 'Na Empório, abrimos aos sábados das 9h às 13h!')
        self.assertEqual(len(calls), 2)

    def test_internal_alert_codes_are_rewritten_in_customer_language(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        for label in ['conflicting_contacts', 'online_withdrawal', 'payment_terms', 'shipping_boundary',
                      'search_policies', r'online\_withdrawal']:
            calls = []

            def model(messages, info):
                calls.append(1)
                if len(calls) == 1:
                    return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
                answer = f'Abrimos às 9h (alerta *{label}*).' if len(calls) == 2 else 'Abrimos às 9h. Posso ajudar com mais alguma coisa?'
                return ModelResponse(parts=[TextPart(answer)])

            with self.subTest(label=label):
                result = create_agent(FunctionModel(model)).run_sync('Qual o horário?',
                                                                   deps=self.context(manual_path=path))
                self.assertEqual(result.output, 'Abrimos às 9h. Posso ajudar com mais alguma coisa?')
                self.assertEqual(len(calls), 3)

    def test_persistent_internal_metadata_is_never_displayed_even_without_tools(self):
        def model(messages, info):
            return ModelResponse(parts=[TextPart('Observação: cuidado online_withdrawal.')])

        result = create_agent(FunctionModel(model)).run_sync('Oi!', deps=self.context())
        self.assertNotIn('online_withdrawal', result.output)
        self.assertNotIn('Observação', result.output)
        self.assertIn('te orientar', result.output)

    def test_shipping_boundary_cannot_be_silently_decided(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['5. Politica de Frete e Entregas',
                               'Frete gratis acima de R$ 500,00.', 'Taxa de R$ 35 abaixo de R$ 500,00.'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'frete'})])
            answer = ('Custa R$ 35 (manual, seção 5, página 1).' if len(calls) == 2 else
                      'O manual não define o caso exato; confirme com a equipe (seção 5, página 1).')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Frete de pedido de R$ 500?', deps=self.context(manual_path=path))
        self.assertIn('não define', result.output)
        self.assertEqual(len(calls), 3)

    def test_conflicting_contact_requires_explicit_warning(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['1. Dados da Empresa', 'Telefone (67) 1111-1111',
                               '7. Atendimento via WhatsApp', 'WhatsApp (67) 2222-2222'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'WhatsApp'})])
            answer = ('WhatsApp: (67) 2222-2222 (seção 7, página 1).' if len(calls) == 2 else
                      'Há telefones divergentes; confirme com a equipe (seções 1 e 7, página 1; seção 7).')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Qual o WhatsApp?', deps=self.context(manual_path=path))
        self.assertIn('divergentes', result.output)
        self.assertEqual(len(calls), 3)

    def test_conflicting_contact_caution_survives_into_a_follow_up(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['1. Dados da Empresa', 'Telefone (67) 1111-1111',
                               '7. Atendimento via WhatsApp', 'WhatsApp (67) 2222-2222'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'telefone'})])
            if len(calls) == 2:
                return ModelResponse(parts=[TextPart(
                    'Há dois números diferentes; confirme com a equipe (seções 1 e 7, página 1).')])
            if len(calls) == 3:
                return ModelResponse(parts=[TextPart('O WhatsApp é (67) 2222-2222.')])
            return ModelResponse(parts=[TextPart(
                'Há números divergentes; confirme o WhatsApp atualizado com a equipe.')])

        agent = create_agent(FunctionModel(model))
        context = self.context(manual_path=path)
        first = agent.run_sync('Quais os telefones da loja?', deps=context)
        second = agent.run_sync('Qual o WhatsApp de vocês?', deps=context,
                                message_history=first.all_messages())
        self.assertIn('divergentes', second.output)
        self.assertEqual(len(calls), 4)

    def test_conflicting_contacts_across_separate_searches_are_detected(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['1. Dados da Empresa', 'Endereço Rua Teste, 123',
                               'Telefone (67) 1111-1111',
                               '7. Atendimento via WhatsApp', 'WhatsApp (67) 2222-2222'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[
                    ToolCallPart('search_policies', {'query': 'endereço'}),
                    ToolCallPart('search_policies', {'query': 'WhatsApp'}),
                ])
            if len(calls) == 2:
                return ModelResponse(parts=[TextPart('Anotei o endereço e o contato.')])
            if len(calls) == 3:
                return ModelResponse(parts=[TextPart('O WhatsApp é (67) 2222-2222.')])
            return ModelResponse(parts=[TextPart(
                'Há dois números divergentes; confirme o WhatsApp atualizado com a equipe.')])

        agent = create_agent(FunctionModel(model))
        context = self.context(manual_path=path)
        first = agent.run_sync('Qual o endereço?', deps=context)
        second = agent.run_sync('Qual o WhatsApp de vocês?', deps=context,
                                message_history=first.all_messages())
        self.assertIn('divergentes', second.output)
        self.assertEqual(len(calls), 4)

    def test_phone_is_not_volunteered_when_the_question_is_about_something_else(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['1. Dados da Empresa', 'Endereço Rua Teste, 123',
                               'Telefone (67) 1111-1111',
                               '2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'endereço horário'})])
            if len(calls) == 2:
                return ModelResponse(parts=[TextPart(
                    'Fica na Rua Teste, 123. Sábado, das 9h às 13h. WhatsApp (67) 1111-1111.')])
            return ModelResponse(parts=[TextPart('Fica na Rua Teste, 123. Sábado, das 9h às 13h.')])

        result = create_agent(FunctionModel(model)).run_sync(
            'Qual o endereço e o horário de sábado?', deps=self.context(manual_path=path))
        self.assertNotIn('1111', result.output)
        self.assertIn('Rua Teste', result.output)
        self.assertEqual(len(calls), 3)

    def test_pix_cannot_be_described_as_applied_to_a_promotional_price(self):
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('product_offer', {'product_id': 1})])
            answer = ('O desconto PIX já está aplicado ao preço promocional: R$ 899,91.' if len(calls) == 2 else
                      'O PIX não acumula com a promoção; o preço final é R$ 899,91.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('E no PIX?', deps=self.context())
        self.assertNotIn('PIX já está aplicado', result.output)
        self.assertIn('não acumula', result.output)
        self.assertEqual(len(calls), 3)

    def test_online_withdrawal_is_not_extended_to_physical_purchases(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['4. Politica de Trocas e Devolucoes',
                               '4.1 Direito de Arrependimento (Compras Online)',
                               'Devolucao em ate 7 dias apos o recebimento.'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'arrependimento'})])
            answer = ('Compra presencial também tem direito de arrependimento (seção 4, página 1).' if len(calls) == 2 else
                      'O arrependimento cobre apenas compras online. Confirme as opções presenciais com a equipe '
                      '(manual, seção 4, página 1).')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Posso devolver?', deps=self.context(manual_path=path))
        self.assertIn('apenas compras online', result.output)
        self.assertEqual(len(calls), 3)

    def test_payment_conflict_is_not_silently_resolved(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['3. Formas de Pagamento', 'Cartao de credito: parcela minima de R$ 100,00.',
                               '3.1 Regras de Parcelamento', 'De 4x a 6x: parcela minima de R$ 80,00.'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'parcelamento'})])
            answer = ('De 4x a 6x, mínimo R$ 80 (seção 3, página 1).' if len(calls) == 2 else
                      'Há condições diferentes na tabela geral e nas faixas detalhadas; '
                      'confirme com a equipe (manual, seção 3, página 1).')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Quais as regras de parcelamento?',
                                                           deps=self.context(manual_path=path))
        self.assertIn('condições diferentes', result.output)
        self.assertEqual(len(calls), 3)

    def test_persistent_ambiguity_uses_only_verified_uncertainty_not_model_claims(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['5. Politica de Frete e Entregas',
                               'Frete gratis acima de R$ 500,00.', 'Taxa de R$ 35 abaixo de R$ 500,00.'])

        def model(messages, info):
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'frete'})])
            return ModelResponse(parts=[TextPart('Você pagará R$ 35 com certeza (seção 5, página 1).')])

        result = create_agent(FunctionModel(model)).run_sync('Meu pedido deu R$ 500. Qual o frete?',
                                                           deps=self.context(manual_path=path))
        self.assertIn('preciso confirmar', result.output)
        self.assertIn('equipe', result.output)
        self.assertNotIn('página', result.output)
        self.assertNotIn('Fonte:', result.output)
        self.assertNotIn('pagará', result.output)
        self.assertNotIn('R$ 35', result.output)

    def test_unsupported_clock_time_is_retried(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário sábado'})])
            answer = ('Sábado: 08:00 às 13:00.' if len(calls) == 2 else
                      'Na Empório, o sábado é das 9h às 13h.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Qual o horário de sábado?',
                                                           deps=self.context(manual_path=path))
        self.assertEqual(result.output, 'Na Empório, o sábado é das 9h às 13h.')
        self.assertEqual(len(calls), 3)

    def test_internal_source_wording_is_retried(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
            answer = ('Segundo nossos documentos internos, abrimos às 9h.' if len(calls) == 2 else
                      'Na Empório, abrimos aos sábados às 9h.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Qual o horário de sábado?',
                                                           deps=self.context(manual_path=path))
        self.assertEqual(result.output, 'Na Empório, abrimos aos sábados às 9h.')
        self.assertEqual(len(calls), 3)

    def test_placeholder_and_invented_phones_are_never_shown(self):
        for answer in ['Ligue para (67) xxxx-xxxx.', 'Ligue para (67) 9999-9999.']:
            calls = []

            def model(messages, info):
                calls.append(1)
                text = answer if len(calls) == 1 else 'Não consigo confirmar o telefone; fale com a equipe da loja.'
                return ModelResponse(parts=[TextPart(text)])

            with self.subTest(answer=answer):
                result = create_agent(FunctionModel(model)).run_sync('Qual o telefone?', deps=self.context())
                self.assertIn('Não consigo confirmar', result.output)
                self.assertNotIn('xxxx', result.output)
                self.assertNotIn('9999', result.output)
                self.assertEqual(len(calls), 2)

    def test_action_promises_are_rewritten(self):
        def model(messages, info):
            return ModelResponse(parts=[TextPart('Vou registrar sua devolução agora.')])

        result = create_agent(FunctionModel(model)).run_sync('Registre minha devolução.', deps=self.context())
        self.assertNotIn('Vou registrar', result.output)
        self.assertIn('só consigo consultar', result.output)
        self.assertIn('equipe', result.output)

    def test_invented_return_logistics_are_retried(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['4. Politica de Trocas e Devolucoes',
                               '4.1 Direito de Arrependimento (Compras Online)',
                               'Devolucao em ate 7 dias corridos.',
                               'O frete de devolucao e por conta da loja.'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'arrependimento'})])
            answer = ('Vamos enviar uma etiqueta dos Correios.' if len(calls) == 2 else
                      'A loja paga o frete de devolução. Fale com a equipe para combinar o envio.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Quem paga o frete de devolução?',
                                                           deps=self.context(manual_path=path))
        self.assertNotIn('etiqueta', result.output.lower())
        self.assertIn('loja paga o frete', result.output)
        self.assertEqual(len(calls), 3)

    def test_missing_manual_cannot_be_turned_into_a_negative_rule(self):
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'horário'})])
            answer = ('Não, não temos funcionamento 24 horas. Não consegui confirmar o horário.' if len(calls) == 2 else
                      'Não consigo confirmar o horário; fale com a equipe da loja.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('Vocês funcionam 24 horas?', deps=self.context())
        self.assertIn('Não consigo confirmar', result.output)
        self.assertNotIn('não funciona 24', result.output)
        self.assertEqual(len(calls), 3)

    def test_pix_percentage_must_match_tool_data(self):
        self.write('products',
                   'product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at',
                   [[1, '999.90', 'Violão A', 1, 'Nylon', 2, 'active', '{}', '2025-01-01'],
                    [2, '1000.00', 'Violão B', 1, 'Aço', 2, 'active', '{}', '2025-01-01']])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('product_offer', {'product_id': 2})])
            answer = ('No PIX são 10% de desconto: R$ 950,00.' if len(calls) == 2 else
                      'No PIX o desconto é de 5%: R$ 950,00.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('E o preço no PIX?', deps=self.context())
        self.assertEqual(result.output, 'No PIX o desconto é de 5%: R$ 950,00.')
        self.assertEqual(len(calls), 3)

    def test_pix_answer_cannot_deny_its_own_discount(self):
        self.write('products',
                   'product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at',
                   [[1, '999.90', 'Violão A', 1, 'Nylon', 2, 'active', '{}', '2025-01-01'],
                    [2, '1000.00', 'Violão B', 1, 'Aço', 2, 'active', '{}', '2025-01-01']])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('product_offer', {'product_id': 2})])
            answer = ('No PIX fica R$ 950,00, sem desconto adicional.' if len(calls) == 2 else
                      'No PIX o desconto é de 5%: R$ 950,00.')
            return ModelResponse(parts=[TextPart(answer)])

        result = create_agent(FunctionModel(model)).run_sync('E no PIX?', deps=self.context())
        self.assertEqual(result.output, 'No PIX o desconto é de 5%: R$ 950,00.')
        self.assertEqual(len(calls), 3)

    def test_follow_up_cannot_invent_logistics_from_memory(self):
        path = self.directory / 'manual.pdf'
        write_manual_pdf(path, ['4. Politica de Trocas e Devolucoes',
                               '4.1 Direito de Arrependimento (Compras Online)',
                               'O frete de devolucao e por conta da loja.'])
        calls = []

        def model(messages, info):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('search_policies', {'query': 'devolução'})])
            if len(calls) == 2:
                return ModelResponse(parts=[TextPart('O frete de devolução é por conta da loja.')])
            if len(calls) == 3:
                return ModelResponse(parts=[TextPart('Vamos enviar uma etiqueta dos Correios.')])
            return ModelResponse(parts=[TextPart('Não tenho esse detalhe; fale com a equipe da loja.')])

        agent = create_agent(FunctionModel(model))
        context = self.context(manual_path=path)
        first = agent.run_sync('Quem paga o frete de devolução?', deps=context)
        second = agent.run_sync('E a etiqueta?', deps=context, message_history=first.all_messages())
        self.assertNotIn('etiqueta', second.output.lower())
        self.assertIn('fale com a equipe', second.output)

    def test_typographic_phone_dashes_and_times_are_normalized(self):
        self.assertEqual(phone_numbers('(67) 3321\u20114500'), {'6733214500'})
        self.assertEqual(phone_numbers('+55 (67) 3321-4500'), {'6733214500'})
        self.assertEqual(clock_times('Sábado das 9h às 13h'), {'09:00', '13:00'})
        self.assertEqual(clock_times('fecha ao meio-dia'), {'12:00'})

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
