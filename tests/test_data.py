import csv
import tempfile
import unittest
from pathlib import Path
from decimal import Decimal

from artefact_ai.data.store import DataError, StoreData


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.write('categories', 'category_id,name,description', [[1, 'Violões', 'Instrumentos']])
        self.write('products', 'product_id,price_brl,name,category_id,description,stock_quantity,status,specs,created_at',
                   [[1, '999.90', 'Violão A', 1, 'Nylon', 2, 'active', '{}', '2025-01-01']])
        self.write('promotions', 'promotion_id,product_id,discount_percent,description,is_active',
                   [[1, 1, 10, 'Oferta', 1]])
        self.write('customers', 'customer_id', [[1]])
        self.write('orders', 'order_id,customer_id,order_date,status,total_brl,payment_method,tracking_code,estimated_delivery',
                   [[1, 1, '2025-01-02', 'pending', '999.90', 'pix', '', '']])
        self.write('order_items', 'order_id,quantity,product_id', [[1, 1, 1]])

    def write(self, table, header, rows):
        with (self.directory / f'{table}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header.split(','))
            writer.writerows(rows)


class StoreTests(StoreFixture, unittest.TestCase):
    def test_preserves_decimal_prices_and_missing_delivery(self):
        store = StoreData.load(self.directory)
        self.assertEqual(store.products[1].price_brl, Decimal('999.90'))
        self.assertIsNone(store.orders[1].estimated_delivery)

    def test_rejects_broken_relationship(self):
        self.write('order_items', 'order_id,quantity,product_id', [[1, 1, 999]])
        with self.assertRaisesRegex(DataError, 'unknown order or product'):
            StoreData.load(self.directory)

    def test_rejects_duplicate_identifiers(self):
        self.write('customers', 'customer_id', [[1], [1]])
        with self.assertRaisesRegex(DataError, 'Duplicate customer'):
            StoreData.load(self.directory)

    def test_rejects_nonfinite_promotion_discount(self):
        self.write('promotions', 'promotion_id,product_id,discount_percent,description,is_active',
                   [[1, 1, 'NaN', 'Oferta', 1]])
        with self.assertRaises(DataError):
            StoreData.load(self.directory)
