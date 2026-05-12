from dataclasses import dataclass
import os
import pickle
import sys
from typing import List

import faiss
import numpy as np
from sqlalchemy import text
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.config import settings
    from src.db import get_engine
else:
    from src.config import settings
    from src.db import get_engine


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)


class SimpleFaissVectorStore:
    def __init__(
        self,
        index: faiss.Index,
        documents: List[Document],
        embeddings: LocalSentenceTransformerEmbeddings,
        reranker: LocalSentenceTransformerEmbeddings,
    ):
        self.index = index
        self.documents = documents
        self.embeddings = embeddings
        self.reranker = reranker
        self.tokenized_docs = [self._tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs) if self.tokenized_docs else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: LocalSentenceTransformerEmbeddings,
        reranker: LocalSentenceTransformerEmbeddings,
    ):
        texts = [d.page_content for d in documents]
        vectors = embeddings.embed_documents(texts)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(index=index, documents=documents, embeddings=embeddings, reranker=reranker)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        q = self.embeddings.embed_query(query)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        dense_k = min(
            max(k * settings.hybrid_candidate_multiplier, k),
            len(self.documents),
        )
        if dense_k == 0:
            return []
        _, dense_indices = self.index.search(q, dense_k)
        dense_ranked = [i for i in dense_indices[0] if 0 <= i < len(self.documents)]

        query_tokens = self._tokenize(query)
        bm25_ranked: List[int] = []
        if self.bm25 and query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_ranked = list(np.argsort(bm25_scores)[::-1][:dense_k])

        if not dense_ranked and not bm25_ranked:
            return []

        # Reciprocal Rank Fusion across dense + lexical rankings.
        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ranked, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranked, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank)

        fused_ranked_indices = [idx for idx, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        candidate_k = min(
            max(k * settings.reranker_candidate_multiplier, k),
            len(fused_ranked_indices),
        )
        candidate_indices = fused_ranked_indices[:candidate_k]
        candidates = [self.documents[i] for i in candidate_indices]

        query_vec = self.reranker.embed_query(query)
        candidate_texts = [doc.page_content for doc in candidates]
        candidate_vecs = self.reranker.embed_documents(candidate_texts)
        scores = np.dot(candidate_vecs, query_vec)
        ranked_indices = np.argsort(scores)[::-1]
        final_k = min(k, len(candidates))
        return [candidates[i] for i in ranked_indices[:final_k]]

    def save_local(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load_local(
        cls,
        path: str,
        embeddings: LocalSentenceTransformerEmbeddings,
        reranker: LocalSentenceTransformerEmbeddings,
    ):
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError("FAISS index not found. Run `python build_schema_index.py` first.")
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        return cls(index=index, documents=documents, embeddings=embeddings, reranker=reranker)


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
WHERE c.TABLE_NAME NOT LIKE '%tmp%'
  AND c.TABLE_NAME NOT LIKE '%temp%'
  AND c.TABLE_NAME NOT LIKE '%bkp%'
  AND c.TABLE_NAME NOT LIKE '%backup%'
  AND c.TABLE_NAME NOT LIKE '%back%'
   AND c.TABLE_NAME NOT LIKE '%test%'
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


def build_or_refresh_vectorstore() -> SimpleFaissVectorStore:
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
    reranker = LocalSentenceTransformerEmbeddings(model_name=settings.reranker_model)
    vectorstore = SimpleFaissVectorStore.from_documents(docs, embeddings, reranker)
    vectorstore.save_local(settings.vector_db_dir)
    return vectorstore


def get_vectorstore() -> SimpleFaissVectorStore:
    embeddings = LocalSentenceTransformerEmbeddings(model_name=settings.embedding_model)
    reranker = LocalSentenceTransformerEmbeddings(model_name=settings.reranker_model)
    return SimpleFaissVectorStore.load_local(settings.vector_db_dir, embeddings, reranker)
