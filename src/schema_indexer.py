from dataclasses import dataclass
from typing import List

from sqlalchemy import text
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.db import get_engine


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@dataclass
class SchemaChunk:
    table_schema: str
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str
    description: str


def fetch_schema_chunks() -> List[SchemaChunk]:
    query = """
    SELECT
        c.TABLE_SCHEMA,
        c.TABLE_NAME,
        c.COLUMN_NAME,
        c.DATA_TYPE,
        c.IS_NULLABLE,
        CAST(ep.value AS NVARCHAR(4000)) AS COLUMN_DESCRIPTION
    FROM INFORMATION_SCHEMA.COLUMNS c
    LEFT JOIN sys.extended_properties ep
      ON ep.major_id = OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME)
     AND ep.minor_id = COLUMNPROPERTY(OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME), c.COLUMN_NAME, 'ColumnId')
     AND ep.name = 'MS_Description'
    ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
    """

    with get_engine().connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    chunks = []
    for row in rows:
        chunks.append(
            SchemaChunk(
                table_schema=row[0],
                table_name=row[1],
                column_name=row[2],
                data_type=row[3],
                is_nullable=row[4],
                description=row[5] or "",
            )
        )
    return chunks


def build_or_refresh_vectorstore() -> Chroma:
    chunks = fetch_schema_chunks()
    docs = []
    for c in chunks:
        content = (
            f"Schema: {c.table_schema}\n"
            f"Table: {c.table_name}\n"
            f"Column: {c.column_name}\n"
            f"Type: {c.data_type}\n"
            f"Nullable: {c.is_nullable}\n"
            f"Description: {c.description}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "schema": c.table_schema,
                    "table": c.table_name,
                    "column": c.column_name,
                },
            )
        )

    embeddings = LocalSentenceTransformerEmbeddings(model_name=settings.embedding_model)
    vectorstore = Chroma(
        collection_name="sql_schema",
        embedding_function=embeddings,
        persist_directory=settings.vector_db_dir,
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass
    vectorstore = Chroma(
        collection_name="sql_schema",
        embedding_function=embeddings,
        persist_directory=settings.vector_db_dir,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def get_vectorstore() -> Chroma:
    embeddings = LocalSentenceTransformerEmbeddings(model_name=settings.embedding_model)
    return Chroma(
        collection_name="sql_schema",
        embedding_function=embeddings,
        persist_directory=settings.vector_db_dir,
    )
