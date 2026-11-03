# Empório da Música — agente de atendimento

Protótipo em Python de um agente de mensagens de texto para a Empório da Música,
loja fictícia de instrumentos musicais de Campo Grande/MS. O atendimento usa
catálogo, pedidos e o manual de políticas fornecidos no desafio, com respostas em
português alinhadas ao tom da loja.

## O que o agente faz

- Responde dúvidas de catálogo, preço, promoção, estoque e PIX.
- Consulta pedidos somente do cliente vinculado à sessão.
- Consulta o PDF de políticas para horários, endereço, pagamentos, frete,
  garantia, privacidade, trocas e devoluções.
- Pede os dados que faltam antes de concluir uma devolução.
- Explicita divergências do manual em vez de escolher uma regra em silêncio.
- Redireciona assuntos fora do escopo e recusa pedidos de dados de outros clientes.
- Não executa compras, devoluções, reembolsos ou atendimentos humanos.

## Arquitetura

- **PydanticAI** com *function calling*: o modelo decide quando consultar as
  ferramentas, mas preços, estoque, pedidos e trechos do manual vêm de código
  determinístico.
- **Ollama Cloud** por HTTP direto, sem Ollama local e sem fallback para outro
  provedor. O modelo de referência é `gpt-oss:120b`; o nome é configurável.
- **Retrieval lexical por capítulo** no PDF, com seção e página preservadas.
  O manual tem cerca de oito páginas; a busca usa normalização de acentos,
  sinônimos e ranking por capítulo completo, mantendo tabelas, exceções e
  continuações entre páginas juntas.
- **CSV como snapshot**: os dados operacionais são carregados a cada execução.
  Não há integração com estoque ou pedidos em tempo real.
- **Histórico em memória por padrão**, com opção de persistência em **SQLite**.
  Não há PostgreSQL, Docker ou pgvector.
- **Typer** para a CLI.

### Por que não RAG vetorial?

Para este material, a busca lexical foi suficiente e mais simples de auditar:
ela mantém os números exatos do manual (prazos, valores, telefones) e não exige
embeddings, ingestão, banco vetorial ou uma chamada extra a um modelo. Na
avaliação local, 30 perguntas answerable ficaram com o capítulo correto entre os
três primeiros; 28 de 28 perguntas de capítulo único ficaram em primeiro lugar, e
as 2 perguntas sem resposta no manual não retornaram capítulo. A busca também é
rápida e determinística.

Isso não prova que retrieval vetorial seria pior em geral: uma avaliação com
embeddings exigiria um conjunto maior de paráfrases e uma baseline medida. Para
um manual pequeno, porém, adicionar pgvector aumentaria a infraestrutura sem
ganho demonstrado.

## Requisitos

- Python 3.14
- [UV](https://docs.astral.sh/uv/)
- Uma chave do [Ollama Cloud](https://ollama.com/settings/keys)
- Os arquivos de dados do desafio em `../data` (ou em outro diretório indicado
  por `DATA_DIR`):
  - CSVs de categorias, produtos, promoções, clientes, pedidos e itens.
  - `políticas_da_loja.pdf`.

Não é necessário Docker.

## Instalação e configuração

```sh
uv sync
cp .env.example .env
```

Edite `.env` e preencha:

```dotenv
OLLAMA_API_KEY=sua_chave
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=gpt-oss:120b
DATA_DIR=../data
```

| Variável | Uso | Padrão |
| --- | --- | --- |
| `OLLAMA_API_KEY` | Chave do Ollama Cloud. Obrigatória para conversar. | vazio |
| `OLLAMA_BASE_URL` | URL compatível com OpenAI; precisa terminar em `/v1`. | `https://ollama.com/v1` |
| `OLLAMA_MODEL` | Modelo usado no chat. O sufixo `:cloud` é removido. | `gpt-oss:120b` |
| `DATA_DIR` | Diretório dos CSVs e do PDF. | `../data` |
| `HISTORY_TURN_LIMIT` | Turnos completos mantidos/carregados. | `10` |
| `HISTORY_DB` | Arquivo SQLite opcional para persistir o histórico. | desativado |

O `.env` é carregado automaticamente pela CLI. Variáveis já presentes no
ambiente têm precedência.

### Dados

Os CSVs e o PDF não fazem parte do repositório. O carregador aceita os nomes
originais do desafio (por exemplo, `desafio_tecnico_ai_eng - products.csv`) ou
nomes simples (`products.csv`), com exatamente um arquivo por tabela.

A identidade do cliente é demonstrativa: `--customer-id` apenas seleciona um ID
para a sessão. O filtro impede devolver pedidos de outro cliente, mas **não é
autenticação**. Em uma integração real, o ID viria de um contexto autenticado.

## Como usar

```sh
uv run artefact-ai --help
uv run artefact-ai chat
uv run artefact-ai chat --message 'Quanto custa o Takamine GD20?'
uv run artefact-ai chat --customer-id 3
```

No chat interativo, `/sair` encerra. O histórico da execução fica em memória.

### Histórico persistente em SQLite

Para retomar uma conversa em outro processo:

```sh
uv run artefact-ai chat \
  --history-db ./artefact_history.sqlite3 \
  --session-id demo
```

Ou defina `HISTORY_DB` no `.env`. Sem `--session-id`, a sessão persistente usa
`local`. O arquivo guarda turnos completos, incluindo chamadas e resultados de
ferramentas; `HISTORY_TURN_LIMIT` limita quantos turnos são carregados e mantidos.
O arquivo é local, contém dados da conversa e não deve ser versionado.

### Consultas diretas, sem modelo

```sh
uv run artefact-ai catalog 'violão' --max-price 1000
uv run artefact-ai product 119
uv run artefact-ai order 1 --customer-id 3
uv run artefact-ai policies 'Qual o horário de sábado?'
```

Os comandos usam o mesmo snapshot de CSVs/PDF e não precisam de chave de modelo.
`order` exige `--customer-id`; a identidade é sempre aplicada no código, nunca
escolhida pelo modelo.

## Decisões de implementação

### Modelo e provedor

O modelo é acessado diretamente em `https://ollama.com/v1` com `OLLAMA_API_KEY`.
Não há Ollama local nem fallback silencioso: se o provedor ou a chave falhar, a
CLI mostra uma mensagem clara. O modelo de referência usado nas conversas de
validação foi `gpt-oss:120b`; a configuração permite trocar sem alterar o código.

### Ferramentas e guardrails

- `search_products`: busca por nome/categoria/orçamento, somente produtos ativos
  com estoque; o orçamento usa o preço promocional antes do PIX.
- `product_offer`: preço de tabela, promoção, PIX, status e estoque por ID.
- `order_summary`: pedido limitado ao `customer_id` da sessão. Pedido inexistente
  e pedido de outro cliente recebem a mesma resposta genérica.
- `search_policies`: retorna capítulos completos do PDF, com seção e página para
  fundamentação interna. As referências só aparecem ao cliente quando ele pede a
  fonte ou quando ajudam a explicar uma divergência.

O validador de saída bloqueia códigos internos de alerta, nomes de ferramentas,
placeholders de telefone, horários que não constam nos trechos, percentuais de
PIX incompatíveis com a ferramenta, promessas de executar ações e etapas
logísticas inventadas. Quando o modelo insiste em uma resposta insegura, a
resposta é substituída por uma orientação de incerteza derivada da fonte, sem
usar outro modelo.

### Regras de dados

- Valores monetários usam `Decimal`; arredondamento é feito por centavo.
- Promoções não são cumulativas. O PIX de 5% não se aplica sobre preço já
  promocional. Se houver mais de uma promoção ativa para o mesmo produto, o
  protótipo assume o maior desconto e, em empate, o menor ID.
- `estimated_delivery` é previsão, não prova de recebimento. O prazo de
  arrependimento não é calculado a partir dela.
- Produtos descontinuados ou sem estoque não são recomendados; a resposta busca
  alternativas disponíveis.

## Ambiguidades do manual

O agente foi instruído a não resolver estas questões silenciosamente:

- **Telefones divergentes:** a seção 1.2 e a capa trazem `(67) 3341-4444`; a
  seção 7 traz `(67) 3321-4500`. O agente apresenta a divergência e orienta
  confirmar o canal atualizado.
- **Parcelamento:** a tabela geral da seção 3 fala em parcela mínima de R$ 100,
  enquanto as faixas da seção 3.1 têm mínimos diferentes. O agente não escolhe
  uma regra sem confirmação.
- **Frete de exatamente R$ 500:** a seção 5.1 define “acima de R$ 500” e “abaixo
  de R$ 500”, mas não o valor exato. O agente não afirma gratuidade nem cobrança.
- **Arrependimento presencial:** a seção 4.1 descreve o direito para compras
  online. O agente pergunta o canal da compra e não estende automaticamente o
  prazo para a loja física, nem afirma que compras presenciais não têm direitos.
- **Promoções sem vigência:** o CSV tem `is_active`, mas não datas de início/fim.
  O snapshot é tratado como estado atual, sem inferir validade pela data.

## Testes e validação

```sh
uv run --locked python -m unittest discover -s tests
```

A suíte cobre configuração, CSVs, consultas, ferramentas, retrieval, validador,
CLI, histórico SQLite e falhas de provedor. Os testes usam fixtures sintéticas e
não dependem de rede, Docker ou credenciais.

Validação opcional com o modelo real (consome cota da API):

```sh
uv run python scripts/evaluate_agent.py --live --output /tmp/opencode/artefact-eval.json
```

O relatório inclui conversas, ferramentas chamadas e checagens automáticas. A
revisão humana continua necessária para tom, clareza e detalhes semânticos.
Na última execução validada: 24 turnos reais, sem falhas nas checagens
determinísticas; no retrieval, 30/30 casos answerable tiveram o capítulo correto
entre os três primeiros, e 28/28 casos de capítulo único ficaram em primeiro.

## Conversas de exemplo

As conversas capturadas estão em [`conversations/`](conversations/). Elas incluem
catálogo/PIX, devolução online, isolamento de pedidos e ambiguidades de política.

## Limitações e próximos passos

- Os CSVs representam um snapshot; não há estoque, preço ou pedido em tempo real.
- A identidade é demonstrativa; não há login, autenticação ou autorização reais.
- O agente é somente leitura: não registra reclamações, devoluções, compras ou
  reembolsos e não transfere para atendentes.
- A busca lexical foi avaliada em um manual pequeno. Um corpus maior, com muitas
  paráfrases, pode justificar embeddings e uma avaliação comparativa.
- O modelo pode variar respostas entre execuções. Os validadores cobrem as
  categorias conhecidas, mas não substituem revisão humana.
- PDFs digitalizados sem texto exigiriam OCR.
- Com mais tempo: autenticação real, persistência de pedidos/tickets, métricas de
  atendimento, testes de regressão com respostas gravadas e uma baseline vetorial.

## Uso de assistentes de código

O projeto foi desenvolvido em pair programming com o agente de código OpenCode,
usando modelos GPT-OSS/DeepSeek para implementação, revisão de testes e análise
dos requisitos. As decisões de arquitetura, as interpretações das ambiguidades e
a revisão final das respostas foram feitas com revisão humana. Ferramentas de IA
foram usadas para acelerar a escrita e a validação, não para substituir a
conferência das fontes.
