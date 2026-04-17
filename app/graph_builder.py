"""
graph_builder.py – Build LangChain GraphDocuments from extracted triples.
"""

import logging
import hashlib
from typing import Optional

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from config import get_neo4j_graph
from kg_extractor import extract_knowledge_graph
from embedding import embed_text
from utils import get_doc_id_from_source, get_chunk_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_TRIPLE_KEYS = {"head", "head_type", "tail", "tail_type", "relation"}

# def _normalize_id(text: str) -> str:
#     return text.lower().replace(" ", "_")

def _build_entity_id(name: str, entity_type: str, project_id: str) -> str:
    raw = f"{project_id}|{entity_type}|{name}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()

def _ensure_constraints() -> None:
    """
    Create uniqueness constraints for Project and Document nodes.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    graph = get_neo4j_graph()
    constraints = [
        """
        CREATE CONSTRAINT project_id IF NOT EXISTS
        FOR (p:Project) REQUIRE p.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """,
    ]
    for constraint in constraints:
        try:
            graph.query(constraint)
            logger.info("Constraint ensured: %s", constraint.strip().splitlines()[1])
        except Exception as e:
            logger.error("Failed to create constraint: %s", e)
            raise

def _create_document_node(
    project_id: str,
    doc_id: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Create Project + Document nodes and OWNS relationship once per document.
    - Uniqueness constraints on project_id and doc_id are enforced at DB level.
    - MERGE guarantees idempotency — safe to call multiple times.
    """
    _meta = metadata or {}
    graph = get_neo4j_graph()
    try:
        graph.query(
            """
            MERGE (p:Project {id: $project_id, project_id: $project_id})

            MERGE (d:Document {id: $doc_id, doc_id: $doc_id})
            ON CREATE SET
                d.project_id = $project_id,
                d.source     = $source,
                d.title      = $title,
                d.created_at = $created_at

            MERGE (p)-[:OWNS]->(d)
            """,
            {
                "project_id": project_id,
                "doc_id":     doc_id,
                "source":     _meta.get("source", "unknown"),
                "title":      _meta.get("title"),
                "created_at": _meta.get("created_at"),
            },
        )
        logger.info("Document node ensured: project_id=%s, doc_id=%s", project_id, doc_id)
    except Exception as e:
        logger.error("Failed to create document node: project_id=%s, doc_id=%s | %s", project_id, doc_id, e)
        raise


def _build_knowledge_graph(
    text: str,
    extracted_info: list[dict],
    project_id: str,
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
    metadata: Optional[dict] = None,
) -> GraphDocument:
    """Build a GraphDocument for a single chunk — does NOT create Document node."""
    node_map: dict = {}
    nodes: list = []
    relationships: list = []
    _meta = metadata or {}

    # ✅ Only Chunk node here — Project + Document already exist
    chunk_props: dict = {
        "id": chunk_id,
        "project_id": project_id,
        "doc_id":     doc_id,
        "chunk_id":   chunk_id,
        "text":       text,
        "source":     _meta.get("source", "unknown"),
        "page":       _meta.get("page"),
    }
    if embedding:
        chunk_props["embedding"] = embedding

    chunk_node = Node(id=chunk_id, type="Chunk", properties=chunk_props)
    nodes.append(chunk_node)
    node_map[f"__chunk__{chunk_id}"] = chunk_node

    graph = get_neo4j_graph()

    # --- Entity nodes & triples ---
    
    mentioned: set[str] = set()  # track MENTIONS edges already added

    for item in extracted_info:
        head_id = _build_entity_id(item["head"], item["head_type"], project_id)
        tail_id = _build_entity_id(item["tail"], item["tail_type"], project_id)

        for nid, item_key, label in [
            (head_id, "head", item["head_type"]),
            (tail_id, "tail", item["tail_type"]),
        ]:
            if nid not in node_map:
                node = Node(
                    id=nid,
                    type=label.capitalize(),
                    properties={"name": item[item_key], "project_id": project_id},
                )
                node_map[nid] = node
                nodes.append(node)
            if nid not in mentioned:
                relationships.append(
                    Relationship(
                        source=chunk_node, target=node_map[nid], type="MENTIONS",
                        properties={"project_id": project_id, "chunk_id": chunk_id},
                    )
                )
                mentioned.add(nid)

        relationships.append(
            Relationship(
                source=node_map[head_id],
                target=node_map[tail_id],
                type=item["relation"].upper(),
                properties={"project_id": project_id, "doc_id": doc_id, "chunk_id": chunk_id},
            )
        )

    source_doc = Document(
        page_content=text,  # full text preserved for RAG retrieval
        metadata={
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source": _meta.get("source", "unknown"),
            "page": _meta.get("page"),
            "title": _meta.get("title"),
            "created_at": _meta.get("created_at"),
        },
    )

    return GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)


def _persist(graph_doc: GraphDocument) -> None:
    """Write a GraphDocument to Neo4j. Separated from building for testability."""
    graph = get_neo4j_graph()
    graph.add_graph_documents([graph_doc], include_source=False, baseEntityLabel=True)
    logger.info("Graph document added to Neo4j.")


# ── Public API ────────────────────────────────────────────────────────────────

def build_knowledge_graph_from_document(
    chunk: Document,
    extracted_info: list[dict],
    project_id: str,
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
) -> GraphDocument:
    graph_doc = _build_knowledge_graph(
        text=chunk.page_content,
        extracted_info=extracted_info,
        project_id=project_id,
        doc_id=doc_id,
        chunk_id=chunk_id,
        embedding=embedding,
        metadata=chunk.metadata,
    )
    _ensure_constraints()  # Ensure constraints before persisting
    try:
        _create_document_node(project_id=project_id, doc_id=doc_id, metadata=chunk.metadata)
    except Exception as e:
        logger.warning("Failed to create document node: project_id=%s, doc_id=%s | %s", project_id, doc_id, e)
    _persist(graph_doc)
    return graph_doc


def build_knowledge_graph_from_string(
    chunk: str,
    extracted_info: list[dict],
    project_id: str,
    doc_id: str,
    chunk_id: str,
    embedding: Optional[list[float]] = None,
    metadata: Optional[dict] = None,
) -> GraphDocument:
    graph_doc = _build_knowledge_graph(
        text=chunk,
        extracted_info=extracted_info,
        project_id=project_id,
        doc_id=doc_id,
        chunk_id=chunk_id,
        embedding=embedding,
        metadata=metadata,
    )
    _ensure_constraints()  # Ensure constraints before persisting
    try:
        _create_document_node(project_id=project_id, doc_id=doc_id, metadata=metadata)
    except Exception as e:
        logger.warning("Failed to create document node: project_id=%s, doc_id=%s | %s", project_id, doc_id, e)
    _persist(graph_doc)
    return graph_doc


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # entity_types = ["person", "school", "award", "company", "product", "characteristic"]
    # relation_types = [
    #     "alumniOf", "worksFor", "hasAward", "isProducedBy",
    #     "hasCharacteristic", "acquired", "hasProject", "isFounderOf", "leads",
    # ]
    entity_types = []
    relation_types = []

    text = """
    Timothy Donald Cook (born November 1, 1960) is an American business executive who has been 
    the chief executive officer (CEO) of Apple since 2011. He had previously been the company's chief 
    operating officer under its co-founder Steve Jobs. Cook joined Apple in March 1998 as a senior vice 
    president for worldwide operations, and then as vice president for worldwide sales and operations. 
    He was appointed chief executive of Apple on August 24, 2011, after Jobs resigned. 
    During his tenure as the chief executive of Apple and while serving on its board of directors, 
    he has advocated for the political reform of international and domestic surveillance, 
    cybersecurity, national manufacturing, and environmental preservation. Since becoming CEO, 
    Cook has also replaced Jobs' micromanagement with a more liberal style and implemented a 
    collaborative culture at Apple.

    Since 2011 when he took over Apple, to 2020, Cook doubled 
    the company's revenue and profit, and the company's market value increased from $348 billion 
    to $1.9 trillion. In 2025, Apple was the second largest technology company by revenue, 
    with US$416 billion. 

    Outside of Apple, Cook has sat on the board of directors of Nike since 2005. 
    He also sits on the board of the National Football Foundation and is a trustee of Duke University, 
    his alma mater. Cook engages in philanthropy; in March 2015 he said he planned to donate his fortune 
    to charity. In 2014, Cook became the first and only chief executive of a Fortune 500 company 
    to publicly come out as gay. 
    
    In October 2014, the Alabama Academy of Honor inducted Cook, 
    who spoke on the state's record of LGBTQ rights. It is the highest honor Alabama gives 
    its citizens. In 2012 and 2021, Cook appeared on the Time 100, Time's annual list of the 100 
    most influential people in the world. As of December 2025, his net worth is estimated at $2.6 billion, 
    according to Forbes. 
    
    Early life and education  Cook was born on November 1, 1960, 
    in the city of Mobile, Alabama. He was baptized in a Baptist church and grew up in the nearby 
    city Robertsdale. His father, Donald Cook, was a shipyard worker. His mother, Geraldine Cook, 
    worked at a pharmacy. 
    
    Cook graduated salutatorian from the public Robertsdale High School in 
    Alabama in 1978. Cook received a Bachelor of Science with a major in industrial engineering 
    from Auburn University in 1982 and a Master of Business Administration from Duke University in 1988.
    """

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    logger.info("Text split into %d chunks.", len(chunks))

    project_id = "demo_project_123"
    doc_id = "Tim_Cook_Bio_2025"

    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d: %s", i, chunk[:100])  # log first 100 chars
        chunk_id = get_chunk_id(doc_id, i)
        extracted = extract_knowledge_graph(
            text=chunk,
            entity_types=entity_types,
            relation_types=relation_types,
        )
        vec = embed_text(chunk)

        # Single call — build + persist happen once
        build_knowledge_graph_from_string(
            chunk=chunk,
            extracted_info=extracted,
            project_id=project_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            embedding=vec,
            metadata={
                "source": "tim_cook_bio.txt",
                "title": "Tim Cook Biography",
                "created_at": "2026-04-16T10:20:00Z",
                "page": 1,
            },
        )
