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

Preencha `GOOGLE_API_KEY` no `.env`. Os modelos são configuráveis por `GOOGLE_EMBEDDING_MODEL` e `GOOGLE_CHAT_MODEL`; consulte a [lista oficial do Gemini](https://ai.google.dev/gemini-api/docs/models) e confirme disponibilidade e cota no seu projeto. O `.env.example` contém sugestões, nunca uma chave. Para o banco fornecido, mantenha `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag`.

Depois da indicação de banco saudável:

```bash
python src/ingest.py
python src/chat.py
```

Pergunte sobre o conteúdo de `document.pdf`. Digite `sair`, `exit` ou `quit` para encerrar; Ctrl+C e EOF também encerram o chat. Perguntas vazias são ignoradas. Para usar outro PDF, ajuste `PDF_PATH` e execute a ingestão novamente.

## Fluxo e decisões

- `src/ingest.py` lê páginas com `PyPDFLoader` e mantém página e posição nos metadados. O splitter usa `chunk_size=1000` e `chunk_overlap=150`; blocos são persistidos em lotes de 64.
- Os IDs incluem a coleção, o hash do PDF e a posição do trecho. Reingerir o mesmo documento atualiza os mesmos registros, evitando duplicatas.
- `src/config.py` acrescenta o hash do nome do modelo à coleção. Isso separa embeddings de modelos diferentes; não é preciso apagar documentos para trocar o modelo. A busca filtra também pelo hash do PDF atual, impedindo mistura com versões antigas. Dados antigos permanecem no banco até limpeza explícita.
- `src/search.py` chama `similarity_search_with_score(..., k=10)` e concatena os trechos no prompt obrigatório. Uma instrução de sistema reforça que PDF e pergunta são dados, não instruções para alterar o comportamento.
- Se não houver trechos, a resposta é determinística: **“Não tenho informações necessárias para responder sua pergunta.”** Quando há trechos, o modelo recebe essa mesma regra. O cumprimento de todas as regras pelo LLM precisa ser avaliado com perguntas reais; o programa não transforma essa instrução em garantia absoluta.
- Cada pergunta usa apenas seu contexto recuperado. O chat não acrescenta histórico de respostas a consultas seguintes.

O PDF precisa conter texto extraível; não há OCR. O PostgreSQL fornecido é um ambiente local de desenvolvimento. O carregador de PDF mantém o pacote recomendado pelo exercício; sua versão instalada emite um aviso de descontinuação de `langchain-community`, documentado pelos mantenedores. As versões estão fixadas em `requirements.txt` para reprodução.

## Validação executada

```bash
PYTHONPATH=src pytest -q
TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag \
  PYTHONPATH=src pytest -q
```

Em 19/10/2026, a segunda execução passou em **6 testes** com PostgreSQL 17/pgvector real no Docker. A suíte verifica limites e posições dos chunks, leitura do PDF, IDs estáveis, busca com dez resultados e filtro por documento, recusa com índice vazio e persistência sem duplicatas após duas ingestões.

O teste de integração cria uma coleção exclusiva, usa vetores determinísticos de teste e remove somente essa coleção ao terminar. Sem `TEST_DATABASE_URL`, ele é ignorado. Esses vetores não estão disponíveis no programa: a execução normal usa o Gemini.

**Pendente:** ingestão com embeddings do Google e perguntas ponta a ponta ao Gemini, pois nenhuma chave foi fornecida. Os testes locais não comprovam qualidade semântica nem consumo de API. A chave deve ficar somente no `.env`, ignorado pelo Git.

Para parar os serviços mantendo os dados, execute `docker compose down`.
