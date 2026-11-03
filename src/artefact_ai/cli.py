"""Local data inspection commands; no database or LLM is required."""

from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Annotated

import typer

from .data.store import DataError, StoreData
from .queries import StoreQueries

app = typer.Typer(no_args_is_help=True, help='Empório da Música: consultas ao snapshot de dados.')
DataDirectory = Annotated[Path, typer.Option('--data-dir', help='Diretório com os CSVs.', envvar='DATA_DIR')]


def queries(directory: Path) -> StoreQueries:
    try:
        return StoreQueries(StoreData.load(directory))
    except DataError as error:
        typer.echo(f'Erro nos dados: {error}', err=True)
        raise typer.Exit(1) from error


def json_value(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def output(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=json_value))


@app.command()
def catalog(
    query: Annotated[str, typer.Argument(help='Nome ou termos de busca.')] = '',
    category: Annotated[str | None, typer.Option('--category')] = None,
    max_price: Annotated[str | None, typer.Option('--max-price', help='Preço máximo, com ponto decimal.')] = None,
    include_unavailable: Annotated[bool, typer.Option('--include-unavailable')] = False,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    data_dir: DataDirectory = Path('../data'),
) -> None:
    """Busque instrumentos por preço promocional (antes de desconto PIX)."""
    budget = None
    if max_price is not None:
        try:
            budget = Decimal(max_price)
            if not budget.is_finite() or budget < 0:
                raise ValueError('Invalid budget')
        except (InvalidOperation, ValueError) as error:
            raise typer.BadParameter('Informe um valor finito e não negativo.', param_hint='--max-price') from error
    result = queries(data_dir).search_products(query, category, budget, not include_unavailable, limit)
    output([asdict(item) for item in result])


@app.command()
def product(
    product_id: Annotated[int, typer.Argument(min=1)],
    data_dir: DataDirectory = Path('../data'),
) -> None:
    """Consulte preço, estoque e promoção de um produto, inclusive indisponível."""
    offer = queries(data_dir).product_offer(product_id)
    if offer is None:
        typer.echo('Produto não encontrado.', err=True)
        raise typer.Exit(1)
    output(asdict(offer))


@app.command()
def order(
    order_id: Annotated[int, typer.Argument(min=1)],
    customer_id: Annotated[int, typer.Option('--customer-id', min=1, help='ID de demonstração; não autentica o cliente.')],
    data_dir: DataDirectory = Path('../data'),
) -> None:
    """Consulte um pedido do cliente informado (modo demonstrativo)."""
    result = queries(data_dir).order_summary(order_id, customer_id)
    if result is None:
        typer.echo('Pedido não encontrado para este cliente.', err=True)
        raise typer.Exit(1)
    output(asdict(result))


def main() -> None:
    app()
