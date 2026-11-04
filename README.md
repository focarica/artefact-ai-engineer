# Empório da Música — Customer Service Agent

A Python text-based customer service prototype for the fictional Empório da Música
instrument store in Campo Grande, Brazil. The agent uses the challenge-provided
CSV dataset and store policy manual to answer customer questions in Brazilian
Portuguese.

## 1. Setup and usage

### Requirements and installation

- Python 3.14
- [UV](https://docs.astral.sh/uv/)
- An [Ollama Cloud](https://ollama.com/settings/keys) API key
- Challenge data, included in `data/`: category, product, promotion, customer,
  order, and order-item CSV files, plus `políticas_da_loja.pdf`.

The application uses the repository's `data/` directory by default. Set
`DATA_DIR` to use a different location. The supplied filenames are preserved;
the loader expects exactly one file for each table. The records are demo data for
the fictional challenge scenario; do not treat their contact details as real.

```sh
uv sync
cp .env.example .env
```

Edit `.env` and provide the model key:

```dotenv
OLLAMA_API_KEY=your_key
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=gpt-oss:120b
DATA_DIR=./data
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `OLLAMA_API_KEY` | Ollama Cloud key; required for agent conversations. | empty |
| `OLLAMA_BASE_URL` | OpenAI-compatible endpoint ending in `/v1`. | `https://ollama.com/v1` |
| `OLLAMA_MODEL` | Chat model. A trailing `:cloud` suffix is removed for direct API calls. | `gpt-oss:120b` |
| `DATA_DIR` | Directory containing the CSVs and policy PDF. | repository `data/` |
| `HISTORY_TURN_LIMIT` | Number of conversation turns retained/loaded per session. | `10` |
| `HISTORY_DB` | Override for the default SQLite file. | `./artefact_history.sqlite3` |

The CLI loads `.env`; environment variables already set take precedence. Docker
is not required.

### Commands

```sh
uv run artefact-ai --help
uv run artefact-ai chat
uv run artefact-ai chat --message 'Quanto custa o Takamine GD20?'
uv run artefact-ai chat --customer-id 3
```

In interactive chat, type `/sair` to exit. Conversation history is persisted to
`./artefact_history.sqlite3` by default. If `--session-id` is omitted, a new
session is created; when the chat exits, the CLI prints a command to resume it.

Use the ID shown on exit to resume a session in the default database:

```sh
uv run artefact-ai chat --session-id SESSION_ID
```

Use `--history-db ./another_file.sqlite3` or set `HISTORY_DB` only when you want
a different file. Sessions are separated by the demo customer ID so order context
is not shared. SQLite stores complete turns, including tool calls and results.
The database contains conversation data and should not be committed. Use
`--no-history` to keep a run in memory only.

Operational queries and policy inspection can also run without calling the model:

```sh
uv run artefact-ai catalog 'violão' --max-price 1000
uv run artefact-ai product 119
uv run artefact-ai order 1 --customer-id 3
uv run artefact-ai policies 'Qual o horário de sábado?'
```

These commands do not require a model key. `order` requires `--customer-id`.

### Tests

```sh
uv run --locked python -m unittest discover -s tests
```

The test suite uses synthetic fixtures and does not require network access,
credentials, or Docker.

## 2. Technical decisions

### Framework and approach

The agent uses **PydanticAI** with function calling. The model decides when to
call a tool, while catalog, order, and policy data are returned by deterministic
Python functions. Customer identity is not a model argument: the application
gets it from the session context and applies it in code.

### Model and provider

The reference model is `gpt-oss:120b`, accessed through Ollama Cloud at
`https://ollama.com/v1` using its OpenAI-compatible API. There is no fallback to
local Ollama or another provider; access failures are reported explicitly. Set
`OLLAMA_MODEL` to change the model.

### Interface

The interface is a **Typer** CLI, sufficient for the prototype's customer
conversations and direct data queries. Separate commands inspect products,
orders, and policy passages without an LLM call.

### Retrieval and response strategy

The eight-page policy manual uses **chapter-level lexical retrieval**, not
embeddings. Text is normalized, synonyms are mapped, and ranking returns complete
chapters so tables, exceptions, and cross-page continuations remain together.
This keeps prices, phone numbers, and time limits auditable without a vector
database or ingestion service.

In the measured retrieval set, the expected chapter appeared in the top three for
30/30 answerable questions; all 28 single-chapter questions ranked the expected
chapter first; and neither of the 2 unanswerable questions returned a chapter.
No vector baseline was measured, so these results support lexical retrieval for
this manual, not a general claim that it is superior.

The prompt defines the store persona, service scope, and when tools should be
consulted. Guardrails validate known factual constraints, protect order data,
and block clearly out-of-scope categories. Policy searches retain section and
page references for grounding, but citations are not shown by default. When the
manual does not support an answer, the agent expresses uncertainty rather than
filling in a rule from memory.

### Persistence

Conversation sessions are stored in **SQLite** by default. Persistence can be
disabled with `--no-history`, or redirected to another file. Complete turns are
stored and sessions are isolated by demo customer ID to avoid mixing order
context. This supports resuming between runs without PostgreSQL or Docker.

### Data handling and assumptions

- Monetary values use `Decimal` and are rounded to cents.
- Budget filtering uses the promotional price before any PIX discount.
- Promotions do not stack. If multiple promotions are active, the largest
  discount wins; ties go to the lowest promotion ID. The 5% PIX discount does not
  apply to an already-promotional price.
- CSVs are snapshots; there is no live connection to inventory, prices, or orders.
- `estimated_delivery` is an estimate, not proof of delivery. It is not used to
  calculate the withdrawal period.
- Missing orders and orders belonging to another customer produce the same
  response, avoiding confirmation that another customer's order exists.
- `--customer-id` is for demonstration, not authentication. A production system
  would get identity from an authenticated context.

### Policy ambiguities

The agent is instructed not to silently choose an interpretation in these cases:

- **Conflicting phone numbers:** the cover/section 1.2 lists `(67) 3341-4444`,
  while section 7 lists `(67) 3321-4500`. The agent should explain the conflict
  and recommend confirming the current contact.
- **Installments:** section 3 gives a R$ 100 minimum installment, while section
  3.1 has different minimums for detailed ranges. The agent asks the team to
  confirm rather than selecting one rule.
- **Shipping for exactly R$ 500:** section 5.1 defines orders above and below
  R$ 500 but not the exact boundary. The agent does not claim it is free or paid.
- **In-store withdrawal:** section 4.1 describes the right for online purchases.
  The agent asks which channel was used and does not conclude that in-store
  purchases have no rights.
- **Promotion validity:** the CSV has `is_active` but no start/end dates. The
  snapshot is treated as current state; validity is not inferred from dates.

### Known limitations and next steps

- Data are snapshots; there is no live inventory, sales, or order integration.
- The agent is read-only: it does not create tickets, transfer to staff, or
  execute purchases, returns, or refunds.
- Lexical retrieval was evaluated on a small manual. A larger corpus or many more
  paraphrases may justify a measured comparison with embeddings.
- Scope categories and guardrails cover known cases; they cannot guarantee
  perfect classification of every possible message. Model responses can vary
  and still require human review.
- Scanned PDFs without a text layer would require OCR.
- With more time: migrate policy retrieval to vector search backed by PostgreSQL,
  benchmark it against the current lexical search, and strengthen guardrails for
  ambiguous cases and new types of scope drift.

## 3. Code assistants

**Codex with GPT models** and **OpenCode with DeepSeek** were used during
implementation, test review, and analysis of the challenge. Architecture choices,
interpretation of policy ambiguities, and final response review received human
oversight. The assistants accelerated development and validation but did not
replace checking the source data.

## 4. Conversation examples

The repository contains five captured conversations in [`conversations/`](conversations/),
covering catalog/PIX pricing, returns, order isolation, policy ambiguities, scope
boundaries, and missing information. The return conversation demonstrates
consulting and applying the store policy manual.
