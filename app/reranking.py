from sentence_transformers import CrossEncoder
from config import get_llm

import logging

logger = logging.getLogger(__name__)

llm = get_llm()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, contexts):
    """Rerank the contexts based on their relevance to the query using a cross-encoder model."""
    pairs = [(query, c["text"]) for c in contexts]
    scores = reranker.predict(pairs)

    for i, c in enumerate(contexts):
        c["rerank_score"] = float(scores[i])

    return sorted(contexts, key=lambda x: x["rerank_score"], reverse=True)

def rerank_with_llm(query, contexts):
    """Rerank the contexts based on their relevance to the query using an LLM."""
    prompt = f"""
    Query: {query}

    Rank the following texts by relevance:

    {contexts}
    """

    return llm.invoke(prompt)