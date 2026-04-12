"""
config.py – Single source of truth for all runtime dependencies.

Every module imports its LLM / Embeddings / Neo4j clients from here.
Swap a model or endpoint in ONE place and the whole pipeline picks it up.

Environment variables (set in .env):
    LITELLM_PROXY_URL       – base URL of your LiteLLM proxy
    LITELLM_PROXY_API_KEY   – API key for the proxy
    LLM_MODEL               – (optional) override default LLM model name
    EMBEDDING_MODEL         – (optional) override default embedding model name
    NEO4J_URI               – bolt/neo4j URI
    NEO4J_USERNAME          – Neo4j username
    NEO4J_PASSWORD          – Neo4j password
    CHUNK_SIZE              – (optional) token chunk size for text splitting (default 500)
    CHUNK_OVERLAP           – (optional) token overlap between chunks (default 30)
    WIKI_TOP_K_RESULTS      – (optional) Wikipedia pages to fetch per query (default 3)
    WIKI_MAX_CHARS_PER_PAGE – (optional) characters to take from each Wikipedia page (default 5000)
"""

import os
import logging
import requests
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
# from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
# from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# Load environment variables from .env file
# load_dotenv(dotenv_path=".env")
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read DATABASE_URL from environment
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------------------------------------------------------------------
# Default model names – override via environment variables
# -------------------------------------------------------------------

DEFAULT_LLM_MODEL       = "gemini/gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "ollama/nomic-embed-text"

LLM_MODEL       = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# ---------------------------------------------------------------------------
# LiteLLM proxy settings
# ---------------------------------------------------------------------------

LITELLM_PROXY_URL     = os.getenv("LITELLM_PROXY_URL", "")
LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY", "")

# ---------------------------------------------------------------------------
# Neo4j settings
# ---------------------------------------------------------------------------

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------

CHUNK_SIZE              = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP           = int(os.getenv("CHUNK_OVERLAP", "30"))
WIKI_TOP_K_RESULTS      = int(os.getenv("WIKI_TOP_K_RESULTS", "3"))
WIKI_MAX_CHARS_PER_PAGE = int(os.getenv("WIKI_MAX_CHARS_PER_PAGE", "5000"))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(model: str = LLM_MODEL, temperature: float = 0) -> ChatOpenAI:
    """
    Return a ChatOpenAI-compatible LLM routed through the LiteLLM proxy.

    Args:
        model:       LiteLLM model string, e.g. "gemini/gemini-2.5-flash".
        temperature: Sampling temperature (0 = deterministic).
    """
    if not LITELLM_PROXY_URL:
        raise EnvironmentError("LITELLM_PROXY_URL is not set in the environment.")
    if not LITELLM_PROXY_API_KEY:
        raise EnvironmentError("LITELLM_PROXY_API_KEY is not set in the environment.")

    logger.debug("Creating LLM: model=%s  base_url=%s", model, LITELLM_PROXY_URL)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=LITELLM_PROXY_API_KEY,
        base_url=LITELLM_PROXY_URL,
    )

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class LiteLLMEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings that call the LiteLLM proxy's
    /v1/embeddings endpoint.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model    = model
        self.base_url = LITELLM_PROXY_URL
        self.api_key  = LITELLM_PROXY_API_KEY

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:
        data = self._post({"model": self.model, "input": text})
        logger.debug("embed_query: model=%s  text_len=%d", self.model, len(text))
        return data["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = self._post({"model": self.model, "input": texts})
        logger.debug("embed_documents: model=%s  n=%d", self.model, len(texts))
        return [d["embedding"] for d in data["data"]]


@lru_cache(maxsize=1)
def get_embeddings(model: str = EMBEDDING_MODEL) -> LiteLLMEmbeddings:
    """
    Return a cached LiteLLMEmbeddings instance.
    Pass a different *model* string to get an uncached instance.
    """
    return LiteLLMEmbeddings(model=model)

# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_neo4j_graph() -> Neo4jGraph:
    """Return a Neo4jGraph client using the env-configured credentials."""
    return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


def get_neo4j_vector(
    index_name: str = "vector",
    node_label: str = "Chunk",
    text_node_property: str = "text",
    embedding_node_property: str = "embedding",
) -> Neo4jVector:
    """Return a Neo4jVector store backed by the configured embeddings."""
    return Neo4jVector(
        embedding=get_embeddings(),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        node_label=node_label,
        text_node_property=text_node_property,
        embedding_node_property=embedding_node_property,
    )