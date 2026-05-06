# Agentic SQL RAG Assistant (LangGraph + Streamlit)

This project builds an agentic SQL assistant for SQL Server:

- Schema-aware RAG over DB metadata (`INFORMATION_SCHEMA` + extended properties)
- Direct open-source embeddings via `SentenceTransformer`
- Direct FAISS vector index (`faiss`) with local persistence
- Hybrid retrieval: FAISS dense + BM25 lexical + RRF fusion
- Bi-encoder reranking before passing context to SQL generation
- LangGraph pipeline: retrieve schema -> generate SQL -> execute -> answer
- Retrieval evaluation: Precision@K, Recall@K, MRR

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure environment

Create `.env` (or copy from `.env.example`) and set values read by `os.getenv` in `src/config.py`.

Key variables:

- `DB_TYPE`, `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE`
- `DB_DRIVER`, `DB_JDBC_URL`, `DB_JDBC_DRIVER`
- `HF_MODEL_ID`, `HUGGINGFACEHUB_API_TOKEN`
- `EMBEDDING_MODEL`, `RERANKER_MODEL`
- `HYBRID_CANDIDATE_MULTIPLIER`, `RRF_K`, `RERANKER_CANDIDATE_MULTIPLIER`
- `VECTOR_DB_DIR`, `MAX_SQL_ROWS`
- Optional tracing: `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`

## 3) Build schema index

```bash
python build_schema_index.py
```

## 4) Run retrieval evaluation

```bash
python -m eval.evaluate_retrieval
```

## 5) Run Streamlit app

```bash
streamlit run app.py
```

The app allows you to refresh schema index, ask DB questions, review generated SQL, and inspect results.

## Run notes

- Prefer module-style runs from repo root when possible.
- If DB connectivity fails, verify network/VPN access and SQL Server ODBC driver installation.
