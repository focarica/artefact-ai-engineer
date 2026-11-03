from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ToolReturnPart

from artefact_ai.manual import ManualError, PolicyCaution, PolicyManual, PolicySearchResult

PAGES = [
    '''Empório da Música Manual de Políticas e Procedimentos
1. Sobre a Empório da Música
1.2 Dados da Empresa
Endereço Rua Teste, 123 — Campo Grande/MS
Telefone / WhatsApp (67) 1111-1111
2. Horário de Funcionamento
Segunda a sexta 09:00 às 18:00
Sábado 09:00 às 13:00
Domingo e feriados fechado
Página 1''',
    '''3. Formas de Pagamento
Cartão de crédito até 12x, parcela mínima R$ 100.
3.1 Regras de Parcelamento
Parcelamento de 4x a 6x, parcela mínima de R$ 80.
4. Política de Trocas e Devoluções
4.1 Direito de Arrependimento (Compras Online)
Devolução em até 7 dias corridos após o recebimento.
Reembolso em até 10 dias úteis na mesma forma de pagamento.
4.4 Itens Não Elegíveis para Troca
Instrumentos com personalização e produtos com aviso de venda final.
5. Política de Frete e Entregas
5.1 Entregas locais
Frete grátis acima de R$ 500 e taxa de R$ 35 abaixo de R$ 500.
5.2 Entregas para Outras Cidades
Página 2''',
    '''Empório da Música Manual de Políticas e Procedimentos
PAC entrega em 5 a 12 dias úteis. SEDEX em 2 a 5 dias úteis.
7. Atendimento via WhatsApp
WhatsApp (67) 2222-2222
7.2 Fluxo de Atendimento Padrão
    1. Saudação: cumprimente o cliente.
    2. Entendimento: identifique a necessidade.
Página 3''',
    '''9. Privacidade e Proteção de Dados
Os dados pessoais (nome, telefone, e-mail,
endereço) são utilizados exclusivamente para processamento e entrega de pedidos.
Página 4''',
]


def write_manual_pdf(path: Path, lines: list[str]) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page[NameObject('/Resources')] = DictionaryObject({
        NameObject('/Font'): DictionaryObject({NameObject('/F1'): DictionaryObject({
            NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
            NameObject('/BaseFont'): NameObject('/Helvetica'),
        })}),
    })
    commands = [b'BT /F1 12 Tf 50 750 Td']
    for line in lines:
        escaped = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        commands.append(f'({escaped}) Tj 0 -20 Td'.encode('latin-1'))
    commands.append(b'ET')
    stream = DecodedStreamObject()
    stream.set_data(b' '.join(commands))
    page.replace_contents(stream)
    writer.write(path)


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.manual = PolicyManual.from_pages(PAGES, source='políticas.pdf')

    def test_alert_identifiers_stay_internal_when_tool_results_are_serialized(self):
        caution = PolicyCaution('conflicting_contacts', 'Confirme o contato atualizado com a equipe.')
        result = PolicySearchResult('ok', cautions=(caution,))
        payload = to_jsonable_python(result)
        self.assertNotIn('code', payload['cautions'][0])
        self.assertNotIn('conflicting_contacts', str(payload))
        self.assertIn('Confirme o contato', payload['cautions'][0]['message'])
        self.assertEqual(caution.code, 'conflicting_contacts')
        tool_message = ToolReturnPart('search_policies', result).model_response_str()
        self.assertNotIn('conflicting_contacts', tool_message)
        self.assertNotIn('"code"', tool_message)

    def test_address_is_not_confused_with_privacy_mentions(self):
        passages = self.manual.search('Qual o endereço da loja?', limit=1)
        self.assertEqual(passages[0].section, '1. Sobre a Empório da Música')
        self.assertEqual(passages[0].page, 1)
        self.assertEqual(passages[0].source, 'políticas.pdf')
        self.assertIn('Rua Teste', passages[0].text)

    def test_accents_and_paraphrases_are_normalized(self):
        passages = self.manual.search('Que horas abre no sabado?', limit=1)
        self.assertEqual(passages[0].section, '2. Horário de Funcionamento')

    def test_return_rule_includes_exclusions(self):
        passage = self.manual.search('Me arrependi, posso devolver?', limit=1)[0]
        self.assertIn('4.1 Direito de Arrependimento', passage.text)
        self.assertIn('4.4 Itens Não Elegíveis', passage.text)
        self.assertIn('recebimento', passage.text)

    def test_installment_rules_stay_with_general_payment_table(self):
        passage = self.manual.search('Posso parcelar em seis vezes?', limit=1)[0]
        self.assertIn('parcela mínima R$ 100', passage.text)
        self.assertIn('parcela mínima de R$ 80', passage.text)

    def test_cross_page_continuation_keeps_chapter_and_page_numbers(self):
        passages = self.manual.search('Frete para outras cidades pelo SEDEX', limit=1)
        self.assertEqual([p.page for p in passages], [2, 3])
        self.assertTrue(all(p.section == '5. Política de Frete e Entregas' for p in passages))
        self.assertIn('SEDEX', passages[1].text)

    def test_conflicting_contacts_are_both_retrievable(self):
        text = '\n'.join(p.text for p in self.manual.search('Qual o WhatsApp?'))
        self.assertIn('(67) 1111-1111', text)
        self.assertIn('(67) 2222-2222', text)

    def test_numbered_instructions_are_not_chapter_headings(self):
        passages = self.manual.search('WhatsApp', limit=1)
        self.assertEqual(passages[0].section, '7. Atendimento via WhatsApp')
        self.assertIn('1. Saudação', passages[0].text)
        self.assertNotIn('Página 3', passages[0].text)

    def test_unrelated_query_and_stop_words_have_no_results(self):
        self.assertEqual(self.manual.search('fotossíntese de dinossauros'), ())
        self.assertEqual(self.manual.search('qual a politica da loja'), ())

    def test_invalid_search_parameters_are_rejected(self):
        for query, limit in [('', 1), (' ' * 3, 1), ('x' * 501, 1), ('frete', 0), ('frete', 6)]:
            with self.subTest(query=query, limit=limit), self.assertRaises(ValueError):
                self.manual.search(query, limit=limit)

    def test_contacts_shipping_boundary_and_payment_ambiguities_are_reported(self):
        warnings = self.manual.cautions(self.manual.passages)
        self.assertEqual({warning.code for warning in warnings},
                         {'conflicting_contacts', 'shipping_boundary', 'payment_terms', 'online_withdrawal'})
        self.assertEqual(next(w.amount_brl for w in warnings if w.code == 'shipping_boundary'), '500')

    def test_unambiguous_chapter_has_no_warning(self):
        self.assertEqual(self.manual.cautions(self.manual.search('horário', limit=1)), ())

    def test_real_pdf_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manual.pdf'
            write_manual_pdf(path, ['2. Horario de Funcionamento', 'Sabado 09:00 as 13:00'])
            result = PolicyManual.from_pdf(path).search('horario')[0]
            self.assertIn('Sabado', result.text)
            self.assertEqual(result.page, 1)

    def test_missing_and_image_only_pdfs_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manual.pdf'
            with self.assertRaisesRegex(ManualError, 'Não foi possível ler'):
                PolicyManual.from_pdf(path)
            writer = PdfWriter()
            writer.add_blank_page(width=600, height=800)
            writer.write(path)
            with self.assertRaisesRegex(ManualError, 'OCR'):
                PolicyManual.from_pdf(path)

    def test_password_protected_pdf_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manual.pdf'
            writer = PdfWriter()
            writer.add_blank_page(width=600, height=800)
            writer.encrypt('secret')
            writer.write(path)
            with self.assertRaisesRegex(ManualError, 'senha'):
                PolicyManual.from_pdf(path)
