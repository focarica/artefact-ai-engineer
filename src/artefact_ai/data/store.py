"""Load CSV snapshots and reject invalid rows or broken relationships."""

import csv
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, TypeVar

from .models import Category, Order, OrderItem, Product, Promotion

T = TypeVar('T')


class DataError(ValueError):
    """A source file is missing, ambiguous or invalid."""


def integer(value: str, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError('Integer below minimum')
    return result


def money(value: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError('Expected a finite, nonnegative decimal')
    return result


def product(row: dict[str, str]) -> Product:
    status = row['status']
    if status not in {'active', 'coming_soon', 'discontinued'}:
        raise ValueError('Unknown product status')
    specs = json.loads(row['specs'])
    if not isinstance(specs, dict):
        raise ValueError('Product specs must be an object')
    return Product(integer(row['product_id']), money(row['price_brl']), row['name'],
                   integer(row['category_id']), row['description'],
                   integer(row['stock_quantity'], 0), status, specs,
                   date.fromisoformat(row['created_at']))


def promotion(row: dict[str, str]) -> Promotion:
    discount = money(row['discount_percent'])
    if discount > 100 or row['is_active'] not in {'0', '1'}:
        raise ValueError('Invalid promotion')
    return Promotion(integer(row['promotion_id']), integer(row['product_id']),
                     discount, row['description'], row['is_active'] == '1')


def order(row: dict[str, str]) -> Order:
    if row['status'] not in {'cancelled', 'confirmed', 'delivered', 'pending', 'shipped'}:
        raise ValueError('Unknown order status')
    return Order(integer(row['order_id']), integer(row['customer_id']),
                 date.fromisoformat(row['order_date']), row['status'],
                 money(row['total_brl']), row['payment_method'],
                 row['tracking_code'] or None,
                 date.fromisoformat(row['estimated_delivery']) if row['estimated_delivery'] else None)


def read_rows(directory: Path, table: str, parser: Callable[[dict[str, str]], T]) -> list[T]:
    # Support both the supplied filenames and simpler names in fixtures/distributions.
    paths = [p for p in directory.glob('*.csv')
             if p.stem == table or p.stem.endswith(f' - {table}')]
    if len(paths) != 1:
        raise DataError(f'Expected exactly one {table}.csv in {directory}')
    try:
        with paths[0].open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise DataError(f'Invalid CSV header: {paths[0].name}')
            results = []
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('Malformed CSV row')
                    results.append(parser(row))
                except (ValueError, TypeError, KeyError, InvalidOperation) as error:
                    raise DataError(f'Invalid {table} record at CSV line {reader.line_num}') from error
            return results
    except (OSError, UnicodeError, csv.Error) as error:
        raise DataError(f'Cannot read {paths[0].name}') from error


def keyed(records: list[T], key: Callable[[T], int]) -> dict[int, T]:
    result = {key(record): record for record in records}
    if len(result) != len(records):
        raise DataError('Duplicate record identifier')
    return result


@dataclass(frozen=True)
class StoreData:
    categories: dict[int, Category]
    products: dict[int, Product]
    promotions: dict[int, Promotion]
    orders: dict[int, Order]
    order_items: tuple[OrderItem, ...]
    customer_ids: frozenset[int]

    @classmethod
    def load(cls, directory: Path) -> 'StoreData':
        categories = keyed(read_rows(directory, 'categories', lambda r: Category(
            integer(r['category_id']), r['name'], r['description'])), lambda r: r.category_id)
        products = keyed(read_rows(directory, 'products', product), lambda r: r.product_id)
        promotions = keyed(read_rows(directory, 'promotions', promotion), lambda r: r.promotion_id)
        orders = keyed(read_rows(directory, 'orders', order), lambda r: r.order_id)
        items = tuple(read_rows(directory, 'order_items', lambda r: OrderItem(
            integer(r['order_id']), integer(r['quantity']), integer(r['product_id']))))
        customer_ids = read_rows(directory, 'customers', lambda r: integer(r['customer_id']))
        if len(set(customer_ids)) != len(customer_ids):
            raise DataError('Duplicate customer identifier')
        if any(p.category_id not in categories for p in products.values()):
            raise DataError('Product references an unknown category')
        if any(p.product_id not in products for p in promotions.values()):
            raise DataError('Promotion references an unknown product')
        if any(o.customer_id not in customer_ids for o in orders.values()):
            raise DataError('Order references an unknown customer')
        if any(i.order_id not in orders or i.product_id not in products for i in items):
            raise DataError('Order item references an unknown order or product')
        if len({(i.order_id, i.product_id) for i in items}) != len(items):
            raise DataError('Duplicate order item')
        return cls(categories, products, promotions, orders, items, frozenset(customer_ids))
