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
uso pela futura camada de persistência; a aplicação ainda não carrega essa variável.
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

Pacote UV, PydanticAI, Typer e infraestrutura do banco preparados. Agente, CLI de chat,
conexão Python, tabelas de histórico e indexação vetorial ainda serão implementados.
Os CSVs continuam sendo a fonte das consultas operacionais.

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
