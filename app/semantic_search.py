import logging
from collections import defaultdict

from config import get_neo4j_graph, get_neo4j_vector

logger = logging.getLogger(__name__)


def _get_entities_for_chunks(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """Single batched query to fetch entities for all chunks at once."""
    if not chunk_ids:
        return {}

    graph = get_neo4j_graph()
    try:
        result = graph.query("""
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {id: chunk_id})-[:MENTIONS]->(e)
        RETURN chunk_id, e.name AS name, e.id AS id, labels(e) AS labels
        """, {"chunk_ids": chunk_ids})
    except Exception as e:
        logger.warning(f"Entity batch fetch failed: {e}")
        return {}

    grouped = defaultdict(list)
    for row in result:
        grouped[row["chunk_id"]].append({
            "name": row["name"],
            "id": row["id"],
            "labels": row["labels"]
        })
    return dict(grouped)


def semantic_search_with_score(
    query: str,
    k: int = 5,
    project_id: str = "demo_project_123"
) -> list[dict]:

    vectorstore = get_neo4j_vector()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter={"project_id": project_id}
    )

    # Batch-fetch all entities in one round trip
    chunk_ids = [
        doc.metadata["chunk_id"]
        for doc, _ in results
        if doc.metadata.get("chunk_id")
    ]
    entities_by_chunk = _get_entities_for_chunks(chunk_ids)

    structured = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id")
        logger.info(f"chunk_id={chunk_id} score={score:.4f}")

        # Prefer live graph entities; fall back to whatever was stored in metadata
        entities = (
            entities_by_chunk.get(chunk_id)
            or doc.metadata.get("entities", [])
        )

        structured.append({
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
            "entities": entities,
        })

    return structured

def _show_results(results: list[dict]):
    for idx, item in enumerate(results):
        logger.info(f"Result {idx+1}:")
        logger.info(f"Score: {item['score']:.4f}")
        logger.info(f"Text: {item['text'][:200]}...")
        logger.info(f"Metadata: {item['metadata']}")
        logger.info(f"Entities: {item['entities']}")
        logger.info("-" * 40)

# --- Example usage ---
if __name__ == "__main__":

    query = "Which universities did Cook graduate from, and in which years??"
    results = semantic_search_with_score(query=query, k=5)

    _show_results(results)
    

    query2 = "At which companies has Cook served as a director?"
    results2 = semantic_search_with_score(query=query2, k=5)

    _show_results(results2)


