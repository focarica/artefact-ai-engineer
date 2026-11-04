# Empório da Música — agente de atendimento

Protótipo em Python de um agente de mensagens de texto para a loja fictícia
Empório da Música, em Campo Grande/MS. O projeto usa os CSVs e o manual de
políticas fornecidos no desafio para responder dúvidas de clientes.

## 1. Como configurar e executar

### Requisitos e instalação

- Python 3.14
- [UV](https://docs.astral.sh/uv/)
- Uma chave do [Ollama Cloud](https://ollama.com/settings/keys)
- Dados do desafio, incluídos em `data/`: CSVs de categorias, produtos,
  promoções, clientes, pedidos e itens, além de `políticas_da_loja.pdf`.

O carregador usa `data/` no repositório por padrão. Para outro local, configure
`DATA_DIR`. Os arquivos mantêm os nomes originais do desafio e há exatamente um
arquivo por tabela. Os registros são material de demonstração do cenário
fictício; não use os contatos como dados reais.

```sh
uv sync
cp .env.example .env
```

Edite `.env` e configure a chave do modelo:

```dotenv
OLLAMA_API_KEY=sua_chave
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=gpt-oss:120b
DATA_DIR=./data
```

| Variável | Uso | Padrão |
| --- | --- | --- |
| `OLLAMA_API_KEY` | Chave do Ollama Cloud; necessária para conversar com o agente. | vazio |
| `OLLAMA_BASE_URL` | Endpoint compatível com OpenAI, terminado em `/v1`. | `https://ollama.com/v1` |
| `OLLAMA_MODEL` | Modelo do chat. O sufixo `:cloud` é removido para a API direta. | `gpt-oss:120b` |
| `DATA_DIR` | Diretório dos CSVs e do PDF. | `data/` no repositório |
| `HISTORY_TURN_LIMIT` | Turnos mantidos/carregados por sessão. | `10` |
| `HISTORY_DB` | Sobrescreve o arquivo SQLite padrão. | `./artefact_history.sqlite3` |

A CLI carrega `.env`; variáveis já definidas no ambiente têm precedência. Não é
necessário instalar ou iniciar Docker.

### Comandos

```sh
uv run artefact-ai --help
uv run artefact-ai chat
uv run artefact-ai chat --message 'Quanto custa o Takamine GD20?'
uv run artefact-ai chat --customer-id 3
```

No chat interativo, `/sair` encerra. Por padrão, o histórico é salvo em
`./artefact_history.sqlite3`. Se `--session-id` não for informado, uma sessão
nova é criada; ao sair, a CLI mostra o comando para retomá-la.

Use o ID mostrado no encerramento para continuar uma sessão no arquivo padrão:

```sh
uv run artefact-ai chat --session-id ID_DA_SESSAO
```

Só use `--history-db ./outro_arquivo.sqlite3` ou configure `HISTORY_DB` se
quiser outro arquivo. As sessões são separadas pelo cliente demonstrativo para
não misturar o contexto dos pedidos. O SQLite guarda turnos completos, incluindo
chamadas e resultados de ferramentas. O arquivo contém dados da conversa e não
deve ser versionado. `--no-history` desativa a persistência para aquela execução.

Consultas operacionais e inspeção do manual também podem ser feitas sem chamar o
modelo:

```sh
uv run artefact-ai catalog 'violão' --max-price 1000
uv run artefact-ai product 119
uv run artefact-ai order 1 --customer-id 3
uv run artefact-ai policies 'Qual o horário de sábado?'
```

Esses comandos não precisam de chave de modelo. `order` exige `--customer-id`.

### Testes

```sh
uv run --locked python -m unittest discover -s tests
```

A suíte usa fixtures sintéticas e não requer rede, credenciais ou Docker.

## 2. Decisões técnicas

### Framework e abordagem

O agente usa **PydanticAI** com *function calling*. O modelo escolhe quando
consultar ferramentas, mas os valores de catálogo, pedidos e trechos de política
vêm de funções determinísticas. A identidade de cliente não é argumento do
modelo: é obtida do contexto da sessão e aplicada pelo código.

### Modelo e provedor

O modelo de referência é `gpt-oss:120b`, acessado via Ollama Cloud em
`https://ollama.com/v1`. A integração usa a API compatível com OpenAI diretamente.
Não existe fallback para Ollama local ou outro provedor: falhas de acesso são
apresentadas ao operador de forma explícita. O modelo pode ser alterado por
`OLLAMA_MODEL`.

### Interface

A interface é uma CLI feita com **Typer**, opção suficiente para demonstrar as
conversas e as consultas exigidas no protótipo. Também há comandos diretos para
inspecionar produtos, pedidos e políticas sem usar o LLM.

### Retrieval e estratégia de resposta

O manual, com oito páginas, usa busca **lexical por capítulo**, não embeddings.
O texto é normalizado, sinônimos são associados e o ranking retorna capítulos
completos, preservando tabelas, exceções e continuações entre páginas. Isso mantém
valores, telefones e prazos auditáveis sem banco vetorial nem infraestrutura de
ingestão.

Na avaliação realizada, 30/30 perguntas answerable tiveram o capítulo esperado
entre os três primeiros resultados; 28/28 perguntas de capítulo único ficaram em
primeiro lugar, e 2/2 perguntas sem resposta no manual não retornaram capítulo.
Não foi medida uma baseline vetorial, portanto esses dados mostram suficiência
para este manual, não superioridade geral da busca lexical.

O prompt define a persona, o escopo de atendimento e quando consultar ferramentas.
Guardrails validam regras factuais conhecidas, protegem dados de pedidos e
impedem respostas para categorias claras fora do escopo. A busca de políticas
inclui seção e página para fundamentação; referências não são mostradas por padrão.
Se o manual não sustenta uma resposta, o agente declara a incerteza em vez de
completar a regra por memória.

### Persistência

O histórico de sessões é salvo em **SQLite** por padrão, com opção de desativá-lo
(`--no-history`) ou escolher outro arquivo. Cada sessão guarda os turnos completos
e é isolada pelo cliente demonstrativo. Isso oferece retomada entre execuções sem
exigir PostgreSQL ou Docker.

### Tratamento dos dados e suposições

- Valores monetários usam `Decimal`, com arredondamento por centavo.
- O preço usado para filtrar orçamento é o preço promocional, antes do PIX.
- Promoções não acumulam. Se houver várias promoções ativas, vale o maior
  desconto; em empate, o menor ID. O PIX de 5% não se aplica sobre preço já
  promocional.
- Os CSVs são snapshots: não há conexão a estoque, preço ou pedido em tempo real.
- `estimated_delivery` é previsão, não comprovação de recebimento. Não é usada
  para calcular o prazo de arrependimento.
- Pedidos inexistentes e pedidos de outro cliente têm a mesma resposta, evitando
  confirmar a existência de pedidos alheios.
- `--customer-id` é demonstrativo, não autenticação. Em produção, a identidade
  teria de vir de um contexto autenticado.

### Ambiguidades do manual

O agente foi instruído a não escolher silenciosamente uma interpretação nestes
casos:

- **Telefones divergentes:** capa/seção 1.2 trazem `(67) 3341-4444`; seção 7
  traz `(67) 3321-4500`. O agente deve explicitar a divergência e recomendar
  confirmar o contato atualizado.
- **Parcelamento:** a seção 3 dá parcela mínima de R$ 100, enquanto a seção 3.1
  detalha faixas com mínimos diferentes. O agente pede confirmação.
- **Frete de exatamente R$ 500:** a seção 5.1 define valores acima e abaixo,
  mas não o limite exato. O agente não afirma gratuidade nem cobrança.
- **Arrependimento presencial:** a seção 4.1 descreve o direito para compras
  online. O agente pergunta o canal da compra e não conclui que compras
  presenciais não tenham direitos.
- **Vigência das promoções:** o CSV contém `is_active`, mas não datas de início e
  término. O snapshot é tratado como estado atual, sem inferir validade pela data.

### Limitações conhecidas e próximos passos

- A busca lexical foi avaliada em um manual pequeno. Um corpus maior ou com muitas
  paráfrases pode justificar uma comparação com embeddings.
- As categorias de escopo e os guardrails tratam casos conhecidos; não garantem
  a classificação perfeita de toda mensagem possível. Respostas do modelo variam
  e ainda requerem revisão humana.
- PDFs sem camada de texto precisariam de OCR.
- Com mais tempo: migrar a busca do manual para retrieval vetorial com PostgreSQL,
  avaliar a qualidade contra a busca lexical atual e reforçar os guardrails para
  cobrir melhor casos ambíguos e novos tipos de desvio de escopo.

## 3. Uso de assistentes de código

Foram usados **Codex com modelos GPT** e **OpenCode com DeepSeek** como assistentes
de código durante a implementação, a revisão dos testes e a análise do enunciado.
As decisões de arquitetura, as interpretações das ambiguidades e a revisão final
das respostas tiveram revisão humana. Os assistentes aceleraram o desenvolvimento
e a validação, sem substituir a conferência das fontes.

## 4. Exemplos de interação

O repositório inclui cinco conversas capturadas em [`conversations/`](conversations/),
cobrindo catálogo/PIX, devolução, isolamento de pedidos, ambiguidades de política,
limites de escopo e falhas de informação. A conversa de devolução demonstra a
consulta e aplicação das regras do manual.
