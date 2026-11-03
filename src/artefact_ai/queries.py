"""Deterministic store queries usable by the CLI and future agent tools."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import unicodedata

from .data.store import StoreData

CENT = Decimal('0.01')


def normalized(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(c))


@dataclass(frozen=True)
class ProductOffer:
    product_id: int
    name: str
    category: str
    status: str
    stock_quantity: int
    available: bool
    original_price_brl: Decimal
    price_brl: Decimal
    discount_percent: Decimal
    promotion: str | None
    pix_price_brl: Decimal


@dataclass(frozen=True)
class OrderLine:
    product_name: str
    quantity: int


@dataclass(frozen=True)
class OrderSummary:
    order_id: int
    status: str
    order_date: date
    total_brl: Decimal
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    items: tuple[OrderLine, ...]


class StoreQueries:
    def __init__(self, data: StoreData):
        self.data = data

    def product_offer(self, product_id: int) -> ProductOffer | None:
        product = self.data.products.get(product_id)
        if product is None:
            return None
        promotions = [p for p in self.data.promotions.values()
                      if p.product_id == product_id and p.is_active]
        # Multiple active promotions do not stack; choose the highest discount.
        promotion = max(promotions, key=lambda p: (p.discount_percent, -p.promotion_id), default=None)
        discount = promotion.discount_percent if promotion else Decimal(0)
        price = (product.price_brl * (1 - discount / 100)).quantize(CENT, rounding=ROUND_HALF_UP)
        pix_price = price if promotion else (price * Decimal('0.95')).quantize(CENT, rounding=ROUND_HALF_UP)
        return ProductOffer(
            product.product_id, product.name, self.data.categories[product.category_id].name,
            product.status, product.stock_quantity,
            product.status == 'active' and product.stock_quantity > 0,
            product.price_brl, price, discount,
            promotion.description if promotion else None, pix_price,
        )

    def search_products(
        self, query: str = '', category: str | None = None,
        max_price_brl: Decimal | None = None, available_only: bool = True,
        limit: int = 10,
    ) -> list[ProductOffer]:
        """Filter by current promotional price, excluding payment-specific PIX discounts."""
        if not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        if max_price_brl is not None and (not max_price_brl.is_finite() or max_price_brl < 0):
            raise ValueError('max_price_brl must be finite and nonnegative')
        tokens = normalized(query).split()
        result = []
        for product in self.data.products.values():
            offer = self.product_offer(product.product_id)
            assert offer is not None
            searchable = normalized(f'{product.name} {product.description} {offer.category}')
            if not all(token in searchable for token in tokens):
                continue
            if category and normalized(category) not in normalized(offer.category):
                continue
            if available_only and not offer.available:
                continue
            if max_price_brl is not None and offer.price_brl > max_price_brl:
                continue
            result.append(offer)
        return sorted(result, key=lambda item: (item.price_brl, item.product_id))[:limit]

    def order_summary(self, order_id: int, customer_id: int) -> OrderSummary | None:
        """customer_id must come from a trusted context, not arbitrary LLM arguments.

        This ownership check is not authentication. The prototype CLI uses demo IDs.
        Missing orders and orders belonging to other customers return the same result.
        """
        order = self.data.orders.get(order_id)
        if order is None or order.customer_id != customer_id:
            return None
        items = tuple(OrderLine(self.data.products[item.product_id].name, item.quantity)
                      for item in self.data.order_items if item.order_id == order_id)
        return OrderSummary(order.order_id, order.status, order.order_date, order.total_brl,
                            order.payment_method, order.tracking_code, order.estimated_delivery, items)
