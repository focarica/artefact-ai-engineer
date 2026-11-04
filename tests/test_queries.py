import unittest
from dataclasses import replace
from decimal import Decimal

from artefact_ai.queries import StoreQueries
from artefact_ai.data.models import Promotion
from artefact_ai.data.store import StoreData
from test_data import StoreFixture


class QueryTests(StoreFixture, unittest.TestCase):
    def queries(self):
        return StoreQueries(StoreData.load(self.directory))

    def test_active_promotion_does_not_stack_pix(self):
        offer = self.queries().product_offer(1)
        self.assertEqual(offer.price_brl, Decimal('899.91'))
        self.assertEqual(offer.pix_price_brl, offer.price_brl)
        self.assertEqual(offer.original_price_brl, Decimal('999.90'))

    def test_inactive_promotion_uses_regular_price_and_pix(self):
        self.write('promotions', 'promotion_id,product_id,discount_percent,description,is_active',
                   [[1, 1, 50, 'Expired', 0]])
        offer = self.queries().product_offer(1)
        self.assertEqual(offer.price_brl, Decimal('999.90'))
        self.assertEqual(offer.pix_price_brl, Decimal('949.91'))
        self.assertIsNone(offer.promotion)

    def test_search_ignores_accents_and_uses_promotional_budget(self):
        queries = self.queries()
        self.assertEqual(len(queries.search_products('VIOLAO', max_price_brl=Decimal('900'))), 1)
        self.assertEqual(queries.search_products(max_price_brl=Decimal('899')), [])

    def test_category_filter_matches_singular_and_plural(self):
        queries = self.queries()
        self.assertEqual(len(queries.search_products(category='violão')), 1)
        self.assertEqual(len(queries.search_products(category='violões')), 1)

    def test_unavailable_products_are_not_recommended(self):
        queries = self.queries()
        queries.data.products[1] = replace(queries.data.products[1], stock_quantity=0)
        self.assertEqual(queries.search_products(), [])
        self.assertFalse(queries.product_offer(1).available)
        self.assertEqual(len(queries.search_products(available_only=False)), 1)
        queries.data.products[1] = replace(queries.data.products[1], stock_quantity=5, status='discontinued')
        self.assertEqual(queries.search_products(), [])

    def test_highest_active_discount_wins_without_stacking(self):
        queries = self.queries()
        queries.data.promotions[2] = Promotion(2, 1, Decimal(20), 'Better', True)
        self.assertEqual(queries.product_offer(1).price_brl, Decimal('799.92'))

    def test_order_ownership_and_missing_order_share_result(self):
        queries = self.queries()
        self.assertIsNone(queries.order_summary(1, 2))
        self.assertIsNone(queries.order_summary(99, 1))
        summary = queries.order_summary(1, 1)
        self.assertEqual(summary.items[0].quantity, 1)
        self.assertIsNone(summary.estimated_delivery)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self.queries().search_products(max_price_brl=Decimal('NaN'))
