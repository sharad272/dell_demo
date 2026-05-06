# Agentic SQL RAG Assistant (LangGraph + Streamlit)

This project builds an agentic RAG chat assistant for SQL Server:

- Uses SQL Server schema metadata as retrieval context
- Uses open-source embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Uses local vector database (Chroma)
- Uses OpenAI OSS model through Hugging Face (`openai/gpt-oss-120b`)
- Uses LangGraph pipeline to retrieve schema -> generate SQL -> execute -> answer
- Includes retrieval evaluation: Precision@K, Recall@K, MRR

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Environment variables

Credentials and model settings are loaded from `.env` using `os.getenv` in `src/config.py`.

## 3) Build schema vector index

```bash
python build_schema_index.py
```

## 4) Run evaluation

```bash
python -m eval.evaluate_retrieval
```

## 5) Start Streamlit app

```bash
streamlit run app.py
```

## Notes

- For Hugging Face hosted inference, set `HUGGINGFACEHUB_API_TOKEN` in your environment if required.
- If DB connection fails, verify ODBC driver and network/VPN access.
