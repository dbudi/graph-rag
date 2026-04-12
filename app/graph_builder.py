"""
graph_builder.py – Build LangChain GraphDocuments from extracted triples.

Neo4j credentials come from config.py – no os.getenv() calls here.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from config import get_neo4j_graph
from kg_extractor import extract_knowledge_graph
from embedding import embed_text
from utils import get_doc_id_from_source, get_chunk_id

# from langchain_community.graphs import Neo4jGraph
# from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_id(text: str) -> str:
    return text.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def _build_knowledge_graph(
    text: str,
    extracted_info: list[dict],
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
    metadata: Optional[dict] = None,
) -> GraphDocument:
    """Internal implementation — works purely on primitives."""
    node_map: dict = {}
    nodes: list = []
    relationships: list = []
    _meta = metadata or {}

    doc_node = Node(
        id=doc_id, 
        type="Document", 
        properties={
            "doc_id": doc_id,
            "source": _meta.get("source", "unknown"),
            "title":  _meta.get("title"),
        }
    )
    nodes.append(doc_node)
    node_map[doc_id] = doc_node

    chunk_props: dict = {
        "chunk_id": chunk_id, 
        "text": text,
        "source":   _meta.get("source", "unknown"),
        "page":     _meta.get("page"),
        "title":    _meta.get("title"),}
    if embedding:
        chunk_props["embedding"] = embedding

    chunk_node = Node(id=chunk_id, type="Chunk", properties=chunk_props)
    nodes.append(chunk_node)
    node_map[chunk_id] = chunk_node

    relationships.append(
        Relationship(source=doc_node, target=chunk_node, type="HAS_CHUNK", properties={})
    )

    for item in extracted_info:
        head_id = _normalize_id(item["head"])
        if head_id not in node_map:
            head_node = Node(
                id=head_id,
                type=item["head_type"].capitalize(),
                properties={"name": item["head"]},
            )
            node_map[head_id] = head_node
            nodes.append(head_node)
        else:
            head_node = node_map[head_id]

        tail_id = _normalize_id(item["tail"])
        if tail_id not in node_map:
            tail_node = Node(
                id=tail_id,
                type=item["tail_type"].capitalize(),
                properties={"name": item["tail"]},
            )
            node_map[tail_id] = tail_node
            nodes.append(tail_node)
        else:
            tail_node = node_map[tail_id]

        relationships.append(Relationship(
            source=head_node, target=tail_node,
            type=item["relation"].upper(), properties={},
        ))
        relationships.append(Relationship(source=chunk_node, target=head_node, type="MENTIONS", properties={}))
        relationships.append(Relationship(source=chunk_node, target=tail_node, type="MENTIONS", properties={}))

    source_doc = Document(
        page_content=text[:300],  # include a snippet of the original text for reference
        metadata={
            "doc_id":   doc_id,
            "chunk_id": chunk_id,
            "source":   _meta.get("source", "unknown"),
            "page":     _meta.get("page"),
            "title":    _meta.get("title"),
        },
    )

    graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)
    try:
        graph = get_neo4j_graph()
        graph.add_graph_documents([graph_doc])
        logger.info("Graph document added to Neo4j.")
    except Exception as e:
        logger.error("Failed to add graph document to Neo4j: %s", e)
        raise
    return graph_doc


# ── Public API ────────────────────────────────────────────────────────────────

def build_knowledge_graph_from_document(
    chunk: Document,
    extracted_info: list[dict],
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
) -> GraphDocument:
    """
    Convert a text chunk + extracted triples into a LangChain GraphDocument.

    Args:
        chunk:          The source Document chunk.
        extracted_info: List of triple dicts from extract_knowledge_graph().
        doc_id:         Stable hash-based document identifier.
        chunk_id:       Stable hash-based chunk identifier.
        embedding:      Optional pre-computed embedding vector for the chunk.

    Returns:
        A GraphDocument ready to be added to Neo4j via graph.add_graph_documents().
    """
    return _build_knowledge_graph(
        text=chunk.page_content,
        extracted_info=extracted_info,
        doc_id=doc_id,
        chunk_id=chunk_id,
        embedding=embedding,
        metadata=chunk.metadata,      # Document carries its own metadata
    )


def build_knowledge_graph_from_string(
    chunk: str,
    extracted_info: list[dict],
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
    metadata: Optional[dict] = None,  # caller supplies metadata explicitly
) -> GraphDocument:
    """
    Convert a text chunk + extracted triples into a LangChain GraphDocument.

    Args:
        chunk:          The source text chunk.
        extracted_info: List of triple dicts from extract_knowledge_graph().
        doc_id:         Stable hash-based document identifier.
        chunk_id:       Stable hash-based chunk identifier.
        embedding:      Optional pre-computed embedding vector for the chunk.

    Returns:
        A GraphDocument ready to be added to Neo4j via graph.add_graph_documents().
    """
    return _build_knowledge_graph(
        text=chunk,
        extracted_info=extracted_info,
        doc_id=doc_id,
        chunk_id=chunk_id,
        embedding=embedding,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    entity_types   = ["person", "school", "award", "company", "product", "characteristic"]
    relation_types = ["alumniOf", "worksFor", "hasAward", "isProducedBy",
                      "hasCharacteristic", "acquired", "hasProject", "isFounderOf", "leads"]

    chunk_text = """
    Timothy Donald Cook (born November 1, 1960) is an American business executive who has been 
    the chief executive officer (CEO) of Apple since 2011. He had previously been the company's chief 
    operating officer under its co-founder Steve Jobs. Cook joined Apple in March 1998 as a senior vice 
    president for worldwide operations, and then as vice president for worldwide sales and operations. 
    He was appointed chief executive of Apple on August 24, 2011, after Jobs resigned. 
    During his tenure as the chief executive of Apple and while serving on its board of directors, 
    he has advocated for the political reform of international and domestic surveillance, 
    cybersecurity, national manufacturing, and environmental preservation. Since becoming CEO, 
    Cook has also replaced Jobs' micromanagement with a more liberal style and implemented a 
    collaborative culture at Apple. Since 2011 when he took over Apple, to 2020, Cook doubled 
    the company's revenue and profit, and the company's market value increased from $348 billion 
    to $1.9 trillion. In 2025, Apple was the second largest technology company by revenue, 
    with US$416 billion. Outside of Apple, Cook has sat on the board of directors of Nike since 2005. 
    He also sits on the board of the National Football Foundation and is a trustee of Duke University, 
    his alma mater. Cook engages in philanthropy; in March 2015 he said he planned to donate his fortune 
    to charity. In 2014, Cook became the first and only chief executive of a Fortune 500 company 
    to publicly come out as gay. In October 2014, the Alabama Academy of Honor inducted Cook, 
    who spoke on the state's record of LGBTQ rights. It is the highest honor Alabama gives 
    its citizens. In 2012 and 2021, Cook appeared on the Time 100, Time's annual list of the 100 
    most influential people in the world. As of December 2025, his net worth is estimated at $2.6 billion, 
    according to Forbes. Early life and education  Cook was born on November 1, 1960, 
    in the city of Mobile, Alabama. He was baptized in a Baptist church and grew up in the nearby 
    city Robertsdale. His father, Donald Cook, was a shipyard worker. His mother, Geraldine Cook, 
    worked at a pharmacy. Cook graduated salutatorian from the public Robertsdale High School in 
    Alabama in 1978. Cook received a Bachelor of Science with a major in industrial engineering 
    from Auburn University in 1982 and a Master of Business Administration from Duke University in 1988.
    """
    extracted = extract_knowledge_graph(
        text=chunk_text,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    logger.info("Extracted: %s", extracted)

    doc_id   = get_doc_id_from_source(chunk_text)
    chunk_id = get_chunk_id(doc_id, 0)
    vec      = embed_text(chunk_text)  # type: ignore

    graph_doc_from_string = build_knowledge_graph_from_string(
        chunk=chunk_text,
        extracted_info=extracted,
        doc_id=doc_id,
        chunk_id=chunk_id,
        embedding=vec,
        metadata={},
    )

    graph = get_neo4j_graph()
    graph.add_graph_documents([graph_doc_from_string])
    logger.info("Graph document added to Neo4j.")
