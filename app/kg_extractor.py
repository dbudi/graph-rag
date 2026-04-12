from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import logging
import time
from typing import Optional

from config import get_llm

logger = logging.getLogger(__name__)


class ExtractedInfo(BaseModel):
    head: str = Field(description="extracted first or head entity like Microsoft, Apple, John")
    head_type: str = Field(description="type of the extracted head entity like person, company, etc")
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(description="extracted second or tail entity like Microsoft, Apple, John")
    tail_type: str = Field(description="type of the extracted tail entity like person, company, etc")

# ---------------------------------------------------------------------------
# Few-shot examples (static, shared across calls)
# ---------------------------------------------------------------------------

_EXAMPLES = [
    {
        "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
        "head": "Adam",         "head_type": "person",
        "relation": "worksFor",
        "tail": "Microsoft",    "tail_type": "company",
    },
    {
        "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
        "head": "Adam",         "head_type": "person",
        "relation": "hasAward",
        "tail": "Best Talent",  "tail_type": "award",
    },
    {
        "text": "Microsoft is a tech company that provide several products such as Microsoft Word",
        "head": "Microsoft Word", "head_type": "product",
        "relation": "isProducedBy",
        "tail": "Microsoft",      "tail_type": "company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",   "head_type": "product",
        "relation": "hasCharacteristic",
        "tail": "lightweight app",  "tail_type": "characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",    "head_type": "product",
        "relation": "hasCharacteristic",
        "tail": "accessible offline", "tail_type": "characteristic",
    },
]

# ---------------------------------------------------------------------------
# Prompt templates (built once at module load)
# ---------------------------------------------------------------------------

_parser = JsonOutputParser(pydantic_object=ExtractedInfo)

_system_prompt = PromptTemplate(
    template="""You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Your task is to identify the entities and relations requested with the user prompt, from a given text.
You must generate the output in a JSON containing a list with JSON objects having the following keys: "head", "head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt.
The "head_type" key must contain the type of the extracted head entity which must be one of the types from {entity_types}.
The "relation" key must contain the type of relation between the "head" and the "tail" which must be one of the relations from {relation_types}.
The "tail" key must represent the text of an extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail entity from {entity_types}.
Attempt to extract as many entities and relations as you can.

IMPORTANT NOTES:
- Don't add any explanation and text.""",
        input_variables=["entity_types", "relation_types"],
)

_human_prompt = PromptTemplate(
    template="""Based on the following example, extract entities and relations from the provided text.

Rules to follow when extracting entities and relations:
- Resolve pronouns (he/she/they) to the real entity name.
- Use the following entity types, don't use other entity that is not defined below:
# ENTITY TYPES:
{entity_types}

- Use the following relation types, don't use other relation that is not defined below:
# RELATION TYPES:
{relation_types}

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text, extract entities and relations as in the provided example.\n{format_instructions}\nText: {text}""",
    input_variables=["entity_types", "relation_types", "examples", "text"],
    partial_variables={"format_instructions": _parser.get_format_instructions()},
)

_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=_system_prompt),
    HumanMessagePromptTemplate(prompt=_human_prompt),
])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_knowledge_graph(
    text: str,
    entity_types: list[str],
    relation_types: list[str],
    llm: Optional[BaseChatModel] = None,
) -> list[dict]:
    """
    Extract (head, relation, tail) triples from *text*.

    Args:
        text:           Free-form text to analyse.
        entity_types:   Allowed entity type labels.
        relation_types: Allowed relation type labels.
        llm:            Optional pre-built LangChain chat model.
                        Defaults to config.get_llm() if not supplied.

    Returns:
        List of dicts with keys: head, head_type, relation, tail, tail_type.
    """
    model = llm or get_llm()
    chain = _chat_prompt | model | _parser

    results: list[dict] = []
    try:
        start    = time.perf_counter()
        response = chain.invoke({
            "entity_types":  entity_types,
            "relation_types": relation_types,
            "examples":      _EXAMPLES,
            "text":          text,
        })
        elapsed = time.perf_counter() - start
        logger.info("Extracted triples in %.2fs  (text_len=%d)", elapsed, len(text))

        if isinstance(response, list):
            results.extend(response)
        elif isinstance(response, dict):
            results.append(response)

    except Exception as exc:
        logger.error("KG extraction failed (text_len=%d): %s | text[:100]=%s",
                     len(text), exc, text[:100])

    return results


# --- Example usage ---
if __name__ == "__main__":
    # entity_types = ["person", "school", "award", "company", "product", "characteristic"]
    # relation_types = ["alumniOf", "worksFor", "hasAward", "isProducedBy", "hasCharacteristic", "acquired", "hasProject", "isFounderOf", "leads"]

    entity_types = []
    relation_types = []
    sample_text = (
        "Elon Musk founded SpaceX in 2002. "
        "He also leads Tesla, which produces electric vehicles. "
        "Tesla cars are known for their long range and autopilot feature."
    )

    extracted = extract_knowledge_graph(
        text=sample_text,
        entity_types=entity_types,
        relation_types=relation_types
    )

    for item in extracted:
        logging.info(item)