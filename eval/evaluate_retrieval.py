from dataclasses import dataclass
from typing import List

from src.schema_indexer import get_vectorstore


@dataclass
class EvalExample:
    query: str
    relevant_keywords: List[str]


def precision_at_k(retrieved_docs: List[str], relevant_keywords: List[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    rel = 0
    for doc in top_k:
        if any(key.lower() in doc.lower() for key in relevant_keywords):
            rel += 1
    return rel / k


def recall_at_k(retrieved_docs: List[str], relevant_keywords: List[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    hits = set()
    for key in relevant_keywords:
        for doc in top_k:
            if key.lower() in doc.lower():
                hits.add(key.lower())
    return len(hits) / max(len(set(k.lower() for k in relevant_keywords)), 1)


def mrr(retrieved_docs: List[str], relevant_keywords: List[str]) -> float:
    for idx, doc in enumerate(retrieved_docs, start=1):
        if any(key.lower() in doc.lower() for key in relevant_keywords):
            return 1.0 / idx
    return 0.0


def run_eval():
    vectorstore = get_vectorstore()
    examples = [
        EvalExample(query="Show customer order details", relevant_keywords=["customer", "order"]),
        EvalExample(query="List product inventory status", relevant_keywords=["product", "inventory"]),
        EvalExample(query="Find sales revenue by region", relevant_keywords=["sales", "revenue", "region"]),
    ]

    k = 3
    p_scores, r_scores, mrr_scores = [], [], []
    for ex in examples:
        docs = vectorstore.similarity_search(ex.query, k=3)
        texts = [d.page_content for d in docs]
        p_scores.append(precision_at_k(texts, ex.relevant_keywords, k))
        r_scores.append(recall_at_k(texts, ex.relevant_keywords, k))
        mrr_scores.append(mrr(texts, ex.relevant_keywords))

    print("Retrieval Evaluation")
    print(f"Precision@{k}: {sum(p_scores)/len(p_scores):.4f}")
    print(f"Recall@{k}:    {sum(r_scores)/len(r_scores):.4f}")
    print(f"MRR:           {sum(mrr_scores)/len(mrr_scores):.4f}")


if __name__ == "__main__":
    run_eval()
