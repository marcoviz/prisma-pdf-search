# Prisma PDF

Consulta de documentos pelo terminal com Python, LangChain, Gemini e PostgreSQL/pgvector. O PDF é dividido em trechos de até 1000 caracteres, com sobreposição de 150, e cada pergunta recupera até dez trechos do documento selecionado.

## Como executar

Pré-requisitos: Python 3.12, Docker com Compose e uma chave do Google AI Studio com acesso aos modelos configurados.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

Preencha `GOOGLE_API_KEY` no `.env`. O exemplo usa `gemini-embedding-001` para embeddings e `gemini-3.1-flash-lite` para respostas. Os modelos são configuráveis por `GOOGLE_EMBEDDING_MODEL` e `GOOGLE_CHAT_MODEL`; consulte a [lista oficial do Gemini](https://ai.google.dev/gemini-api/docs/models) e confirme disponibilidade e cota no seu projeto. O `.env.example` contém sugestões, nunca uma chave. Para o banco fornecido, mantenha `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag`.

Depois da indicação de banco saudável:

```bash
python src/ingest.py
python src/chat.py
```

Pergunte sobre o conteúdo de `document.pdf`. Digite `sair`, `exit` ou `quit` para encerrar; Ctrl+C e EOF também encerram o chat. Perguntas vazias são ignoradas. Para usar outro PDF, ajuste `PDF_PATH` e execute a ingestão novamente.

## Fluxo e decisões

- `src/ingest.py` lê páginas com `PyPDFLoader` e mantém página e posição nos metadados. O splitter usa `chunk_size=1000` e `chunk_overlap=150`. Por padrão, persiste lotes de 16 blocos, com 30 segundos entre lotes, para reduzir rajadas de tokens na cota gratuita de embeddings. `INGEST_BATCH_SIZE` aceita de 1 a 64 e `INGEST_BATCH_DELAY_SECONDS` aceita de 0 a 300. Ajuste esses valores à cota do seu projeto; eles não alteram o tamanho ou a sobreposição dos chunks.
- Os IDs incluem a coleção, o hash do PDF e a posição do trecho. Reingerir o mesmo documento atualiza os mesmos registros, evitando duplicatas.
- `src/config.py` acrescenta o hash do nome do modelo à coleção. Isso separa embeddings de modelos diferentes; não é preciso apagar documentos para trocar o modelo. A busca filtra também pelo hash do PDF atual, impedindo mistura com versões antigas. Dados antigos permanecem no banco até limpeza explícita.
- `src/search.py` chama `similarity_search_with_score(..., k=10)` e concatena os trechos no prompt obrigatório. Uma instrução de sistema reforça que PDF e pergunta são dados, não instruções para alterar o comportamento.
- Se não houver trechos, a resposta é determinística: **“Não tenho informações necessárias para responder sua pergunta.”** Quando há trechos, o modelo recebe essa mesma regra. O cumprimento de todas as regras pelo LLM precisa ser avaliado com perguntas reais; o programa não transforma essa instrução em garantia absoluta.
- Cada pergunta usa apenas seu contexto recuperado. O chat não acrescenta histórico de respostas a consultas seguintes.

O PDF precisa conter texto extraível; não há OCR. O PostgreSQL fornecido é um ambiente local de desenvolvimento. O carregador de PDF mantém o pacote recomendado pelo exercício; sua versão instalada emite um aviso de descontinuação de `langchain-community`, documentado pelos mantenedores. As versões estão fixadas em `requirements.txt` para reprodução.

Uma resposta `429 RESOURCE_EXHAUSTED` indica limite do provedor. A ingestão inicial em lotes de 64 chegou a esse limite com a cota gratuita testada; por isso os valores padrão distribuem os 67 chunks do PDF em cinco lotes ao longo de aproximadamente dois minutos. Se houver outros processos consumindo a mesma cota, reduza o lote ou aumente o intervalo. Cada lote que recebe HTTP 429 tem no máximo três novas tentativas, após 30, 60 e 120 segundos, preservando seus IDs. O código reconhece também o erro Gemini encapsulado pelo LangChain, sem depender do texto da mensagem. Erros de autenticação, entrada inválida e banco não são repetidos automaticamente. Se as tentativas se esgotarem, o CLI encerra com erro sem imprimir credenciais; após a recuperação da cota, reexecutar a ingestão não duplica registros. Essas esperas não ampliam a cota diária.

## Validação executada

```bash
PYTHONPATH=src pytest -q
TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag \
  PYTHONPATH=src pytest -q
```

Em 19/10/2026, a suíte original passou em **14 testes**, sem testes ignorados, com PostgreSQL 17/pgvector real no Docker. A cobertura verifica limites e posições dos chunks, leitura do PDF, IDs estáveis, tamanho dos lotes, intervalo entre envios, rejeição de configurações inválidas, busca com dez resultados e filtro por documento, recusa com índice vazio e persistência sem duplicatas após duas ingestões. A única advertência foi a descontinuação do carregador `langchain-community`, já descrita acima. A revisão final acrescenta seis cenários de retry: recuperação de 429 encapsulado, limite de tentativas, falhas 400/403/banco sem retry e proteção contra falso positivo ou ciclo na cadeia de exceções. Esses cenários simulam somente as falhas e a espera; não provocam esgotamento de cota real.

A suíte final foi executada no [GitHub Actions, run 39333812904](https://github.com/marcoviz/prisma-pdf-search/actions/runs/39333812904), em 19/10/2026, sobre o commit `aec683f549de565ff2a6a18ac775e3f19c842a8a`: **20 testes aprovados, zero falhas, zero ignorados e um aviso**, em 3,25 segundos. O job `postgres-integration` usou Python 3.12.15 e PostgreSQL 17/pgvector real; `pip check` também passou. O [registro permanente da execução](evidence/ci-validation-2026-10-19.json) guarda SHA, links e o resultado extraído do log oficial. Essa execução inclui as correções de retry e não faz chamadas ao Gemini.

O teste de integração cria uma coleção exclusiva, usa vetores determinísticos de teste e remove somente essa coleção ao terminar. Sem `TEST_DATABASE_URL`, ele é ignorado. Esses vetores não estão disponíveis no programa: a execução normal usa o Gemini.

A validação com o Google também foi concluída em **19/10/2026**. `gemini-embedding-001` gerou os vetores reais dos **67 chunks** do PDF de 34 páginas, persistidos no PostgreSQL com **3072 dimensões**. Sobre esses mesmos vetores, `gemini-3.1-flash-lite` respondeu corretamente a três perguntas factuais e retornou a frase de recusa exigida nas três perguntas fora do contexto. Cada consulta recuperou dez trechos com filtro pelo hash do PDF. O CLI real passou com entrada vazia, pergunta sobre o ano de fundação da SuperTechIABrazil e encerramento por `sair`.

As respostas, metadados de recuperação, consumo informado pelo modelo e saída do CLI estão em [`evidence/live-validation-2026-10-19.json`](evidence/live-validation-2026-10-19.json). A evidência distingue a execução da ingestão do reteste de consultas: o modelo escolhido inicialmente retornou indisponibilidade 503, então as consultas foram realizadas com o modelo padrão atual, sem reingerir os vetores. São sete verificações reais, sem respostas ou embeddings simulados; essa amostra não garante o comportamento para toda pergunta possível. A chave fica somente no `.env`, ignorado pelo Git.

Para parar os serviços mantendo os dados, execute `docker compose down`.
