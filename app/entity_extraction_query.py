
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

import logging
import json
import re

logger = logging.getLogger(__name__)

llm = get_llm()



def _clean_json(text: str) -> str:
    # Remove ```json ... ``` wrappers
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

_entity_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract named entities from the query."
     "Return JSON list of objects with fields: name, type."
    #  "Types: PERSON, ORG, PLACE, CONCEPT."
     "No explanation."),
    ("human", "{query}")
])

def extract_entities(query: str) -> list[dict]:
    chain = _entity_extraction_prompt | llm
    result = chain.invoke({"query": query})
    raw = result.content
    logger.info(f"LLM response for entity extraction: {raw}")
    cleaned = _clean_json(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        logger.warning(msg="Entity parsing failed", exc_info=True, extra={"result": result})
        return []


# --- Example usage ---
if __name__ == "__main__":
    query = "Which universities did Tim Cook graduate from, and in which years? How long Tim Cook has been CEO of Apple? Is Tim Cook have a Cat who's name is Kitty?"
    entities = extract_entities(query)
    logger.info(f"Extracted entities: {entities}")