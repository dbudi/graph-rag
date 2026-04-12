"""
embedding.py – Convenience functions for embedding text and documents.

LiteLLMEmbeddings and get_embeddings() live in config.py.
This module re-exports them for backward-compatibility and adds
higher-level helpers used by the rest of the pipeline.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# Re-export from config so existing imports still work
from config import get_embeddings, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """Return the embedding vector for a single string."""
    return get_embeddings().embed_query(text)


def embed_documents(documents: List[Document]):
    """
    Embed *documents* and store them in Neo4jVector.

    Returns the Neo4jVector instance (useful for ad-hoc searches).
    """
   
    

    store = Neo4jVector.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    logger.info("Embedded %d documents into Neo4jVector.", len(documents))
    return store


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# Old callers used embed_fn(text) – keep working without changes
embed_fn = embed_text


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    vec = embed_text("testing embedding of a sample chunk of text about Tim Cook and Apple Inc.")
    logger.info("Embedding dimension: %d", len(vec))
