# Artefact AI Engineer

Protótipo de atendimento da Empório da Música em Python, gerenciado com UV.

## Arquitetura planejada

- PydanticAI: agente com ferramentas para consultar os CSVs.
- RAG sobre o PDF de políticas, com embeddings armazenados no pgvector.
- Ollama Cloud como modelo principal e Ollama local como fallback.
- Typer para a interface de terminal.
- PostgreSQL para sessões e mensagens, compartilhando a infraestrutura com o pgvector.

PostgreSQL substitui a proposta inicial de SQLite por familiaridade com a tecnologia,
persistência centralizada e suporte à busca vetorial na mesma base. O custo é exigir
Docker e um serviço adicional para executar o projeto. O modelo de embeddings e sua
dimensionalidade serão definidos na etapa de retrieval.

## Ambiente de desenvolvimento

Requisitos: Python 3.14, UV, Docker Engine e Docker Compose v2.

```sh
uv sync
cp .env.example .env
docker compose up -d --wait
```

O Compose carrega `.env` para configurar o banco. `DATABASE_URL` fica disponível para
uso pela futura camada de persistência; a configuração Python lê essa variável do ambiente do processo.
O banco atende em `localhost:5432`, com dados preservados no volume `postgres_data`.
A imagem fixa PostgreSQL 16 e pgvector 0.8.7.

Confirme a extensão após a inicialização:

```sh
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT extversion FROM pg_extension WHERE extname = 'vector';
SQL
```

Para parar o serviço preservando os dados, execute `docker compose down`.
O script `docker/postgres/001_extensions.sql` roda somente na criação de um volume
vazio. Alterações posteriores no schema serão feitas por migrações versionadas.
Alterar as credenciais em `.env` não altera um banco já inicializado.

## Estado atual

Configuração, carregamento validado dos CSVs e consultas de catálogo/pedidos via
CLI implementados. Infraestrutura PostgreSQL e migração do histórico preparadas,
com execução no banco ainda pendente de validação neste ambiente. Chat com LLM,
persistência Python, extração PDF e busca vetorial ainda serão implementados.

## Uso de IA

Codex foi utilizado para preparar a estrutura inicial, analisar os requisitos e
configurar a infraestrutura local conforme as decisões do autor.

## Configuração da aplicação

`Settings.from_env()` lê `DATABASE_URL`, `DATA_DIR` e `HISTORY_TURN_LIMIT` do
ambiente do processo. Não carrega `.env` automaticamente. Para executar comandos
Python com esse arquivo, use `uv run --env-file .env ...`. O limite de histórico
conta turnos completos, incluindo chamadas e respostas de ferramentas.

## Migrações do banco

Depois de subir o PostgreSQL, aplique o schema do histórico:

```sh
docker compose exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < migrations/001_conversation_history.sql
```

A migração é transacional, registra sua versão e permite reexecução. Sessões têm
UUID; cada turno guarda a lista completa de mensagens PydanticAI em JSONB para
preservar as chamadas de ferramentas. O índice primário permite recuperar os
últimos turnos por sessão. Ainda falta implementar o repositório Python que grava
e recupera essas mensagens. As tabelas vetoriais serão adicionadas após escolher
o modelo de embeddings e sua dimensão.

Testes de configuração (sem banco ou credenciais de modelo):

```sh
uv run python -m unittest discover -s tests
```

## Consultas locais sem banco ou modelo

Execute na raiz do repositório, após `uv sync`. Os CSVs fornecidos pelo desafio
precisam estar em `../data`, ou em um diretório informado com `--data-dir`/`DATA_DIR`.
São aceitos os nomes originais (`desafio_tecnico_ai_eng - products.csv`) ou nomes
simples (`products.csv`), com exatamente um arquivo por tabela.

```sh
uv run artefact-ai --help
uv run artefact-ai catalog --category 'Violões' --max-price 1000
uv run artefact-ai catalog 'Takamine GD20'
uv run artefact-ai product 81
uv run artefact-ai order 1 --customer-id 3
uv run artefact-ai catalog --data-dir /caminho/dos/csvs
```

A saída é JSON com valores monetários como strings decimais. As consultas carregam
os CSVs a cada comando: representam o snapshot fornecido, sem integração com um
sistema de estoque em produção. Todos os seis CSVs são necessários nesta etapa.

- Busca ignora acentos e caixa; ordena por preço e identifica produtos por ID.
- Catálogo recomenda apenas produtos ativos com estoque. Use
  `--include-unavailable` para inspecionar também os demais.
- O orçamento usa preço promocional antes de desconto PIX. PIX aplica 5% apenas
  quando não há promoção ativa. Arredondamento monetário é feito por centavo.
- Se houver múltiplas promoções ativas, assumimos o maior desconto, sem acumular;
  empates usam o menor ID. Esta é uma interpretação documentada do protótipo.
- `--customer-id` serve somente à demonstração. O filtro impede retornar pedidos
  de outro ID, mas não autentica o cliente. Uma integração real deve obter a
  identidade de um contexto autenticado. O futuro agente não poderá escolher
  arbitrariamente esse ID. Dados pessoais e notas internas não são retornados.
- Prazos exibidos são previsões do CSV; não representam recebimento efetivo.

Os testes usam CSVs sintéticos temporários e não dependem dos dados externos,
do Docker, de rede ou de credenciais. Cobrem validação de relações, preços,
promoções, disponibilidade, vínculo dos pedidos e comandos CLI:

```sh
uv run python -m unittest discover -s tests
```
