import json
import unittest

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
