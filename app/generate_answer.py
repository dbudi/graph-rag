from langchain_core.prompts import ChatPromptTemplate
from config import get_llm

def _build_context(contexts: list[dict], max_chars: int = 4000) -> str:
    """Build a context string from the list of contexts, ensuring it does not exceed max_chars."""
    parts = []
    total = 0

    for i, c in enumerate(contexts):
        chunk = f"[{i+1}] {c['text']}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)



_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "Answer the question ONLY using the provided context.\n"
     "If the answer is not in the context, say 'I don't know'.\n"
     "Cite sources using [number]."),
    
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}")
])

def generate_answer(query: str, contexts: list[dict]) -> str:
    """Generate an answer to the query based on the provided contexts."""
    context_text = _build_context(contexts)
    llm = get_llm()

    chain = _qa_prompt | llm

    result = chain.invoke({
        "context": context_text,
        "question": query
    })

    return result.content