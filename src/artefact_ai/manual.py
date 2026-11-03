"""Page-aware lexical retrieval for the small supplied policy manual.

Rank complete numbered chapters rather than isolated sentences so exceptions and
tables stay with their rules. No embeddings, database or additional model needed.
"""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .queries import normalized

STOP_WORDS = frozenset('a ao aos as ate com como da das de do dos e em eu me meu minha na nas no nos '
                       'o os ou para pela pelo por qual quais que se seu sua um uma voces loja manual '
                       'politica politicas gostaria quero saber posso rs'.split())
CONCEPTS = (
    'devolucao devolucoes devolver devolvo devolve arrependi arrependimento reembolso',
    'troca trocas trocar troco',
    'horario horarios funcionamento funciona aberto abre abertura fecha fechamento',
    'endereco localizacao onde fica localizada',
    'parcelamento parcelar parcelas parcela vezes',
    'frete entrega entregas envio envios enviar',
    'garantia garantias',
    'defeito defeitos defeituoso quebrado',
    'pagamento pagamentos pagar pago',
    'telefone telefones contato contatos whatsapp zap',
)
ALIASES = {word: group.split()[0] for group in CONCEPTS for word in group.split()}
CHAPTER = re.compile(r'^(\d{1,2})\.\s+([^\d\W].*)$')
FOOTER = re.compile(r'^Página\s+\d+$', re.IGNORECASE)


class ManualError(ValueError):
    """The policy source is missing, unreadable or has no searchable text."""


def tokens(text: str) -> list[str]:
    return [ALIASES.get(word, word) for word in re.findall(r'[a-z0-9]+', normalized(text))
            if word not in STOP_WORDS and not word.isdigit() and len(word) > 1]


@dataclass(frozen=True)
class ManualPassage:
    source: str
    page: int
    section: str
    text: str


@pydantic_dataclass(frozen=True)
class PolicyCaution:
    # The application needs the identifier, but the model only needs the guidance.
    code: Annotated[Literal['conflicting_contacts', 'shipping_boundary', 'payment_terms', 'online_withdrawal'],
                    Field(exclude=True)]
    message: str
    amount_brl: str | None = None


@dataclass(frozen=True)
class PolicySearchResult:
    status: Literal['ok', 'not_found', 'unavailable']
    passages: tuple[ManualPassage, ...] = ()
    cautions: tuple[PolicyCaution, ...] = ()


class PolicyManual:
    def __init__(self, passages: tuple[ManualPassage, ...]):
        if not passages:
            raise ManualError('O manual não contém texto pesquisável; PDFs de imagem precisam de OCR.')
        self.passages = passages
        self._chapters: dict[str, list[ManualPassage]] = {}
        for passage in passages:
            self._chapters.setdefault(passage.section, []).append(passage)
        self._terms = {section: Counter(tokens(section + ' ' + ' '.join(p.text for p in parts)))
                       for section, parts in self._chapters.items()}

    @classmethod
    def from_pdf(cls, path: Path) -> 'PolicyManual':
        try:
            with path.open('rb') as source:
                reader = PdfReader(source)
                if reader.is_encrypted:
                    raise ManualError('O manual está protegido por senha.')
                pages = [page.extract_text(extraction_mode='layout') or '' for page in reader.pages]
        except (OSError, PyPdfError, ValueError) as error:
            if isinstance(error, ManualError):
                raise
            raise ManualError(f'Não foi possível ler o manual: {path.name}') from error
        return cls.from_pages(pages, source=path.name)

    @classmethod
    def from_pages(cls, pages: list[str], *, source: str) -> 'PolicyManual':
        passages = []
        section = 'Capa e identificação'
        for number, page in enumerate(pages, 1):
            lines: list[str] = []

            def flush() -> None:
                text = '\n'.join(lines).strip()
                if text:
                    passages.append(ManualPassage(source, number, section, text))
                lines.clear()

            for raw_line in page.splitlines():
                line = re.sub(r'\s+', ' ', raw_line).strip()
                if not line or FOOTER.fullmatch(line):
                    continue
                if normalized(line) == 'emporio da musica manual de politicas e procedimentos':
                    continue
                # Layout mode keeps numbered list items indented. Only unindented
                # top-level headings start a chapter; subsections remain intact.
                if raw_line == raw_line.lstrip() and CHAPTER.fullmatch(line):
                    flush()
                    section = line
                lines.append(line)
            flush()
        return cls(tuple(passages))

    def search(self, query: str, *, limit: int = 3) -> tuple[ManualPassage, ...]:
        """Return up to limit complete chapters, including cross-page continuations."""
        if not query.strip() or len(query) > 500:
            raise ValueError('Informe uma consulta de 1 a 500 caracteres.')
        if not 1 <= limit <= 5:
            raise ValueError('limit must be between 1 and 5')
        terms = set(tokens(query))
        if not terms:
            return ()
        count = len(self._terms)
        average_length = sum(sum(words.values()) for words in self._terms.values()) / count or 1
        ranked = []
        for section, words in self._terms.items():
            score = 0.0
            length = sum(words.values())
            for term in terms:
                frequency = words[term]
                if not frequency:
                    continue
                documents = sum(term in other for other in self._terms.values())
                weight = math.log(1 + (count - documents + 0.5) / (documents + 0.5))
                score += weight * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average_length))
                if term in tokens(section):
                    score += weight
                # Table labels such as Endereço and Telefone identify facts more
                # precisely than incidental mentions inside privacy paragraphs.
                if any(ALIASES.get(label[1], label[1]) == term
                       for passage in self._chapters[section] for line in passage.text.splitlines()
                       if (label := re.match(r'^([a-z0-9]+)(?:\s|:)', normalized(line)))):
                    score += 2 * weight
            if score:
                ranked.append((section, score))
        selected = sorted(ranked, key=lambda item: -item[1])[:limit]
        return tuple(passage for section, _ in selected for passage in self._chapters[section])

    @staticmethod
    def cautions(passages: tuple[ManualPassage, ...]) -> tuple[PolicyCaution, ...]:
        """Surface conflicting facts and open boundaries without inventing a resolution."""
        text = normalized('\n'.join(p.text for p in passages))
        cautions = []
        phones = {re.sub(r'\D', '', match) for match in re.findall(r'\(\d{2}\)\s*\d{4,5}-\d{4}', text)}
        if len(phones) > 1:
            cautions.append(PolicyCaution('conflicting_contacts',
                'Há telefones divergentes nos trechos. Não confirme um único número como correto; '
                'explique a dúvida de forma natural e oriente confirmar o contato atualizado com a equipe.'))
        above = {amount.rstrip('.,') for amount in re.findall(r'acima de r\$\s*([\d.,]+)', text)}
        below = {amount.rstrip('.,') for amount in re.findall(r'abaixo de r\$\s*([\d.,]+)', text)}
        for amount in sorted(above & below):
            value = str(Decimal(amount.replace('.', '').replace(',', '.')))
            cautions.append(PolicyCaution('shipping_boundary',
                f'O manual só define valores acima e abaixo de R$ {amount}. Para valor exatamente igual, '
                'não confirme gratuidade nem cobrança: o caso não está definido e exige confirmação da equipe.', value))
        if len(set(re.findall(r'parcela minima(?: de)? r\$\s*([\d.,]+)', text))) > 1:
            cautions.append(PolicyCaution('payment_terms',
                'Há condições gerais e faixas detalhadas com parcelas mínimas diferentes. '
                'Explique os trechos e solicite confirmação em caso de dúvida, sem escolher silenciosamente uma regra.'))
        if 'direito de arrependimento (compras online)' in text:
            cautions.append(PolicyCaution('online_withdrawal',
                'A seção de arrependimento cobre compras online. Não estenda esse direito a compras presenciais '
                'nem invente interpretação do CDC. Para presencial, consulte as demais modalidades de troca '
                'ou oriente confirmar com a equipe.'))
        return tuple(cautions)
