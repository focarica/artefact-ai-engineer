"""Local data inspection and conversational customer service commands."""

from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

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


@app.command()
def chat(
    message: Annotated[str | None, typer.Option('--message', '-m', help='Envie uma mensagem e encerre.')] = None,
    customer_id: Annotated[int | None, typer.Option('--customer-id', min=1, help='Cliente fixo de demonstração; não autentica.')] = None,
    data_dir: Annotated[Path | None, typer.Option('--data-dir', envvar='DATA_DIR')] = None,
) -> None:
    """Converse com o assistente da loja; histórico somente em memória nesta execução."""
    os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
    from httpx import HTTPError
    from openai import APIError
    from pydantic_ai.exceptions import AgentRunError, ModelHTTPError, UserError

    from .agent import StoreContext, create_agent, create_model
    from .chat import ChatSession
    from .config import ChatSettings

    if message is not None and not message.strip():
        raise typer.BadParameter('A mensagem não pode estar vazia.', param_hint='--message')
    try:
        settings = ChatSettings.from_env()
        session = ChatSession(
            create_agent(create_model(settings)), StoreContext(queries(data_dir or settings.data_dir), customer_id),
            turn_limit=settings.history_turn_limit,
        )
    except (ValueError, UserError) as error:
        typer.echo(f'Configuração inválida: {error}', err=True)
        raise typer.Exit(1) from error

    def reply(text: str) -> None:
        try:
            response = session.respond(text)
        except ModelHTTPError as error:
            if error.status_code in {401, 403}:
                detail = 'Verifique OLLAMA_API_KEY e o acesso ao modelo em https://ollama.com/settings/keys'
            elif error.status_code == 404:
                detail = f'Verifique se o modelo {settings.ollama_model} está disponível no Ollama Cloud.'
            else:
                detail = 'O provedor está indisponível ou atingiu seu limite. Tente novamente mais tarde.'
            typer.echo(f'Não foi possível consultar o modelo. {detail}', err=True)
            raise typer.Exit(1) from error
        except (AgentRunError, HTTPError, APIError, TimeoutError) as error:
            typer.echo('Não foi possível concluir a resposta. Verifique o Ollama Cloud e sua conexão e tente novamente.', err=True)
            raise typer.Exit(1) from error
        typer.echo(f'Empório: {response}')

    try:
        if message is not None:
            reply(message)
            return
        typer.echo('Empório da Música — Sua música começa aqui! Digite /sair para encerrar.')
        typer.echo('Histórico temporário; políticas ainda não disponíveis. Consultas usam o snapshot dos CSVs.')
        if customer_id is not None:
            typer.echo(f'Cliente de demonstração: {customer_id} (sem autenticação).')
        while True:
            try:
                text = typer.prompt('Você', prompt_suffix=': ', default='', show_default=False).strip()
            except EOFError:
                break
            if text.casefold() in {'/sair', '/exit'}:
                break
            if text:
                reply(text)
    except (KeyboardInterrupt, typer.Abort):
        pass


def main() -> None:
    load_dotenv(Path.cwd() / '.env', override=False)
    app()
