from urllib.parse import quote_plus
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings


def _resolve_sqlserver_driver() -> str:
    configured = (settings.db_driver or "").strip()
    installed = [d.strip() for d in pyodbc.drivers()]
    if configured and configured in installed:
        return configured

    sqlsrv_candidates = [d for d in installed if "SQL Server" in d]
    if sqlsrv_candidates:
        # Prefer the highest version driver when available.
        return sorted(sqlsrv_candidates)[-1]

    if configured:
        return configured
    return "ODBC Driver 17 for SQL Server"


def _build_connection_string() -> str:
    driver = _resolve_sqlserver_driver()
    user = settings.db_username
    pwd = settings.db_password
    host = settings.db_host
    port = settings.db_port
    database = settings.db_database or "master"

    if settings.db_jdbc_url.startswith("jdbc:sqlserver://"):
        jdbc_body = settings.db_jdbc_url.replace("jdbc:sqlserver://", "", 1)
        host_part = jdbc_body.split(";", 1)[0]
        if ":" in host_part:
            host, port = host_part.split(":", 1)
        else:
            host = host_part

    odbc_parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={host},{port}",
        f"DATABASE={database}",
        f"Encrypt={settings.db_encrypt}",
        f"TrustServerCertificate={settings.db_trust_server_certificate}",
        f"Connection Timeout={settings.db_connection_timeout}",
    ]

    if settings.db_trusted_connection.lower() == "yes":
        odbc_parts.append("Trusted_Connection=yes")
    else:
        odbc_parts.append(f"UID={user}")
        odbc_parts.append(f"PWD={pwd}")

    return f"mssql+pyodbc:///?odbc_connect={quote_plus(';'.join(odbc_parts))}"


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
