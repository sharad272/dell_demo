from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings


def _build_connection_string() -> str:
    driver = quote_plus(settings.db_driver)
    user = quote_plus(settings.db_username)
    pwd = quote_plus(settings.db_password)
    host = settings.db_host
    port = settings.db_port
    database = settings.db_database

    if settings.db_jdbc_url.startswith("jdbc:sqlserver://"):
        jdbc_body = settings.db_jdbc_url.replace("jdbc:sqlserver://", "", 1)
        host_part = jdbc_body.split(";", 1)[0]
        if ":" in host_part:
            host, port = host_part.split(":", 1)
        else:
            host = host_part

    if database:
        return f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{database}?driver={driver}&TrustServerCertificate=yes"
    return f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/master?driver={driver}&TrustServerCertificate=yes"


def get_engine() -> Engine:
    conn_str = _build_connection_string()
    return create_engine(conn_str, pool_pre_ping=True)


def run_sql(sql_query: str, limit: int | None = None) -> list[dict]:
    safe_query = sql_query.strip().rstrip(";")
    if limit:
        safe_query = f"SELECT TOP {limit} * FROM ({safe_query}) AS q"

    with get_engine().connect() as conn:
        result = conn.execute(text(safe_query))
        columns = list(result.keys())
        rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]
