from config import get_neo4j_graph

import logging

logger = logging.getLogger(__name__)

def graph_query(entities: list[dict]) -> list[dict]:
    logger.info(f"Performing graph query for entities: {entities}")
    if not entities:
        logger.warning("No entities provided for graph query.")
        return []

    graph = get_neo4j_graph()

    query = """
    UNWIND $entities AS ent
    MATCH (e {name: ent.name})
    OPTIONAL MATCH (e)-[r]-(related)
    OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
    RETURN 
        e.name AS entity,
        labels(e) AS labels,
        collect(DISTINCT {
            relation: type(r),
            related: related.name
        }) AS relations,
        collect(DISTINCT c.text) AS chunks
    LIMIT 50
    """

    return graph.query(query, {"entities": entities})

# --- Example usage ---
if __name__ == "__main__":
    sample_entities = [
        {"name": "Tim Cook", "type": "PERSON"},
        {"name": "Apple", "type": "ORG"}
    ]
    results = graph_query(sample_entities)
    for res in results:
        print(res)