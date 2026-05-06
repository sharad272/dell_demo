import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    db_type: str = os.getenv("DB_TYPE", "sqlserver")
    db_host: str = os.getenv("DB_HOST", "")
    db_port: str = os.getenv("DB_PORT", "1433")
    db_username: str = os.getenv("DB_USERNAME", "")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_database: str = os.getenv("DB_DATABASE", "")
    db_driver: str = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    db_jdbc_url: str = os.getenv("DB_JDBC_URL", "")
    db_jdbc_driver: str = os.getenv("DB_JDBC_DRIVER", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
    hf_model_id: str = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "sentence-transformers/all-mpnet-base-v2")
    hybrid_candidate_multiplier: int = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "6"))
    reranker_candidate_multiplier: int = int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "4"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "./faiss_index")
    max_sql_rows: int = int(os.getenv("MAX_SQL_ROWS", "200"))


settings = Settings()
