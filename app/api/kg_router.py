from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from graph_builder import build_knowledge_graph_from_string
from kg_extractor import extract_knowledge_graph
from embedding import embed_text

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    title: Optional[str] = None


class BuildKGRequest(BaseModel):
    text: str
    entity_types: list[str]
    relation_types: list[str]
    document_id: str
    chunk_index: int
    metadata: ChunkMetadata


# class NodeOut(BaseModel):
#     id: str
#     type: str
#     properties: dict = Field(default_factory=dict)


# class RelationshipOut(BaseModel):
#     source_id: str
#     target_id: str
#     type: str
#     properties: dict = Field(default_factory=dict)


class GraphDocumentOut(BaseModel):
    nodes: list[str]
    relationships: list[str]
    source_text_snippet: str


class BuildKGResponse(BaseModel):
    status: str                          # "success" | "failed"
    message: str
    graph_document: Optional[GraphDocumentOut] = None


# ---------------------------------------------------------------------------
# Helper: serialize GraphDocument → GraphDocumentOut
# ---------------------------------------------------------------------------

def _serialize_graph_document(graph_doc) -> GraphDocumentOut:
    nodes = [
        # NodeOut(
        #     id=node.id,
        #     type=node.type,
        #     properties={
        #         k: v for k, v in (node.properties or {}).items()
        #         if k != "embedding"          # skip raw vectors — too large
        #     },
        # )
        f"{node.type}:{node.properties.get('name', '')}"  # simple string representation
        for node in graph_doc.nodes
    ]

    relationships = [
        # RelationshipOut(
        #     source_id=rel.source.id,
        #     target_id=rel.target.id,
        #     type=rel.type,
        #     properties=rel.properties or {},
        # )
        f"{rel.source.type}:{rel.source.properties.get('name', '')} -[{rel.type}]-> "
        f"{rel.target.type}:{rel.target.properties.get('name', '')}"
        for rel in graph_doc.relationships
    ]

    return GraphDocumentOut(
        nodes=nodes,
        relationships=relationships,
        source_text_snippet=graph_doc.source.page_content,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/kg/build", response_model=BuildKGResponse)
def build_knowledge_graph(body: BuildKGRequest):
    """
    Extract entities/relations from *text* and return a GraphDocument.

    Steps:
      1. Extract KG triples via the LLM (kg_extractor).
      2. Embed the chunk text.
      3. Build a GraphDocument (graph_builder).
    """
    chunk_id = f"{body.document_id}_{body.chunk_index}"
    metadata = body.metadata.model_dump()

    try:
        # --- 1. KG extraction ---
        extracted_info = extract_knowledge_graph(
            text=body.text,
            entity_types=body.entity_types,
            relation_types=body.relation_types,
        )

        # --- 2. Embedding ---
        embedding = embed_text(body.text)

        # --- 3. Build GraphDocument ---
        graph_doc = build_knowledge_graph_from_string(
            chunk=body.text,
            extracted_info=extracted_info,
            doc_id=body.document_id,
            chunk_id=chunk_id,
            embedding=embedding,
            metadata=metadata,
        )

        return BuildKGResponse(
            status="success",
            message=f"GraphDocument built with {len(graph_doc.nodes)} nodes "
                    f"and {len(graph_doc.relationships)} relationships.",
            graph_document=_serialize_graph_document(graph_doc),
        )

    except Exception as exc:
        logger.exception("Failed to build knowledge graph for doc_id=%s chunk=%d",
                         body.document_id, body.chunk_index)
        raise HTTPException(status_code=500, detail=str(exc))
