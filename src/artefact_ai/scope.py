"""Deterministic input boundaries for the store's customer-service chat."""

import re

from .queries import normalized


OUT_OF_SCOPE_PATTERNS = (
    # A family of programming languages/frameworks and general software-development tasks.
    re.compile(r'\b(?:java|python|javascript|typescript|rust|golang|kotlin|swift|c\+\+|c#|'
               r'spring boot|django|react|angular|vue|fastapi|flask|rails|'
               r'programar|programacao|programador|software|framework\w*|algoritm\w*|'
               r'compilar|depurar|linguagem de programacao|desenvolvimento de sistemas?)\b'),
    # A code-like function call containing an operation, e.g. calculating an expression.
    re.compile(r'\b[a-z_]\w*\s*\([^)]*[=+*/][^)]*\)', re.IGNORECASE),
    # Common requests from other everyday domains, not specific to a named dish.
    re.compile(r'\b(?:receit\w*|culinari\w*|cozinh\w*|comida|prato|sobremesa|bolo|'
               r'assunto juridico|diagnostico medico|aconselhamento financeiro)\b'),
)
ROLEPLAY_PATTERN = re.compile(
    r'\b(?:finja ser|aja como|atue como|assuma (?:o )?papel de|entre no personagem|'
    r'interprete|me trate como)\b'
)


def input_guardrail(message: str) -> str | None:
    """Return a deterministic reply for broad off-scope task categories."""
    text = normalized(message)
    if ROLEPLAY_PATTERN.search(text):
        return ('Posso respeitar sua preferência de tratamento, mas vou manter o atendimento da Empório da Música. '
                'Posso ajudar com instrumentos, catálogo, pedidos ou políticas da loja.')
    if any(pattern.search(text) for pattern in OUT_OF_SCOPE_PATTERNS):
        return ('Sou o assistente da Empório da Música e posso ajudar com instrumentos, catálogo, pedidos e '
                'políticas da loja. Esse assunto fica fora do atendimento por aqui.')
    return None
