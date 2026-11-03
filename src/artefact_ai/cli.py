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

from .banner import WIDTH, banner, styled
from .data.store import DataError, StoreData
from .queries import StoreQueries

app = typer.Typer(no_args_is_help=True, help='Empório da Música: consultas ao snapshot de dados.')
DataDirectory = Annotated[Path, typer.Option('--data-dir', help='Diretório com os CSVs.', envvar='DATA_DIR')]
ManualPath = Annotated[Path | None, typer.Option('--manual-path', envvar='POLICY_MANUAL',
                                               help='PDF de políticas; padrão: políticas_da_loja.pdf no diretório de dados.')]


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


def show_chat_home(customer_id: int | None, session_id: str | None = None) -> None:
    def centered(text: str, color: str, *, bold: bool = False) -> None:
        typer.echo(styled(text.center(WIDTH), color, bold=bold))

    centered('EMPÓRIO DA MÚSICA', typer.colors.BRIGHT_CYAN, bold=True)
    for line in banner():
        typer.echo(line)
    centered('assistente virtual da loja', typer.colors.CYAN)
    centered('Catálogo, pedidos e políticas: estou aqui para ajudar.', typer.colors.WHITE)
    if session_id:
        centered(f'Sessão persistente: {session_id} | histórico SQLite | /sair encerra', typer.colors.YELLOW)
    else:
        centered('Sessão temporária | snapshot dos CSVs | /sair encerra', typer.colors.YELLOW)
    if customer_id is not None:
        centered(f'Cliente demo: {customer_id} (não autenticado)', typer.colors.YELLOW)
    typer.echo(styled('-' * WIDTH, typer.colors.BLUE))


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
def policies(
    query: Annotated[str, typer.Argument(help='Dúvida ou assunto a buscar no manual.')],
    manual_path: ManualPath = None,
    data_dir: DataDirectory = Path('../data'),
    limit: Annotated[int, typer.Option(min=1, max=5, help='Número máximo de capítulos completos.')] = 3,
) -> None:
    """Inspecione os trechos do manual com seção e página, sem banco ou modelo."""
    from .manual import ManualError, PolicyManual

    try:
        manual = PolicyManual.from_pdf(manual_path or data_dir / 'políticas_da_loja.pdf')
        passages = manual.search(query, limit=limit)
    except (ManualError, ValueError) as error:
        typer.echo(f'Erro no manual: {error}', err=True)
        raise typer.Exit(1) from error
    output([asdict(passage) for passage in passages])


@app.command()
def chat(
    message: Annotated[str | None, typer.Option('--message', '-m', help='Envie uma mensagem e encerre.')] = None,
    customer_id: Annotated[int | None, typer.Option('--customer-id', min=1, help='Cliente fixo de demonstração; não autentica.')] = None,
    data_dir: Annotated[Path | None, typer.Option('--data-dir', envvar='DATA_DIR')] = None,
    manual_path: ManualPath = None,
    history_db: Annotated[Path | None, typer.Option('--history-db', envvar='HISTORY_DB',
                                                   help='Arquivo SQLite; padrão: ./artefact_history.sqlite3.')] = None,
    session_id: Annotated[str | None, typer.Option('--session-id',
                                                  help='Identificador da sessão persistente; padrão: local.')] = None,
    no_history: Annotated[bool, typer.Option('--no-history', help='Não persista a conversa em SQLite.')] = False,
) -> None:
    """Converse com o assistente; o histórico é persistido em SQLite por padrão."""
    os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
    from httpx import HTTPError
    from openai import APIError
    from pydantic_ai.exceptions import AgentRunError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded, UserError

    from .agent import StoreContext, create_agent, create_model
    from .chat import ChatSession
    from .config import ChatSettings
    from .history import HistoryError, SQLiteHistory

    if message is not None and not message.strip():
        raise typer.BadParameter('A mensagem não pode estar vazia.', param_hint='--message')
    history = None
    try:
        settings = ChatSettings.from_env()
        directory = data_dir or settings.data_dir
        selected_db = None if no_history else (history_db or settings.history_db)
        if selected_db:
            try:
                history = SQLiteHistory(selected_db)
            except HistoryError as error:
                typer.echo(f'Erro no histórico: {error}', err=True)
                raise typer.Exit(1) from error
            session_id = session_id or 'local'
        session = ChatSession(
            create_agent(create_model(settings)),
            StoreContext(queries(directory), customer_id, manual_path or directory / 'políticas_da_loja.pdf'),
            turn_limit=settings.history_turn_limit, history=history, session_id=session_id,
        )
    except (ValueError, UserError) as error:
        if history is not None:
            history.close()
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
        except UnexpectedModelBehavior as error:
            typer.echo('Não consegui gerar uma resposta consistente com as fontes. '
                       'Reformule a pergunta ou confirme a política com a equipe da loja.', err=True)
            raise typer.Exit(1) from error
        except UsageLimitExceeded as error:
            typer.echo('A consulta atingiu o limite de chamadas. Tente uma pergunta mais específica.', err=True)
            raise typer.Exit(1) from error
        except HistoryError as error:
            typer.echo(f'Erro no histórico: {error}', err=True)
            raise typer.Exit(1) from error
        except (AgentRunError, HTTPError, APIError, TimeoutError) as error:
            typer.echo('Não foi possível concluir a resposta. Verifique o Ollama Cloud e sua conexão e tente novamente.', err=True)
            raise typer.Exit(1) from error
        typer.echo(styled(f'\n  ♫ Empório: {response}\n', typer.colors.BRIGHT_GREEN))

    try:
        if message is not None:
            reply(message)
            return
        show_chat_home(customer_id, session_id if history else None)
        while True:
            try:
                text = typer.prompt(
                    styled('  Você', typer.colors.BRIGHT_MAGENTA, bold=True),
                    prompt_suffix=' › ', default='', show_default=False,
                ).strip()
            except EOFError:
                break
            if text.casefold() in {'/sair', '/exit'}:
                break
            if text:
                reply(text)
    except (KeyboardInterrupt, typer.Abort):
        pass
    finally:
        if history is not None:
            history.close()


def main() -> None:
    load_dotenv(Path.cwd() / '.env', override=False)
    app()
