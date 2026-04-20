def merge_results(semantic_results, graph_results):
    """Merge results from semantic search and graph query, 
    giving priority to semantic search but including graph results for additional context."""
    merged = []

    # From semantic search
    for item in semantic_results:
        merged.append({
            "text": item["text"],
            "source": "semantic",
            "score": item["score"],
            "entities": item["entities"]
        })

    # From graph
    for graph_result in graph_results:
        for chunk in graph_result.get("chunks", []):
            merged.append({
                "text": chunk,
                "source": "graph",
                "score": 0.5,  # default lower confidence
                "entities": [graph_result["entity"]]
            })

    return merged

def deduplicate_context(contexts):
    seen = set()
    unique = []

    for context in contexts:
        key = context["text"]
        if key not in seen:
            seen.add(key)
            unique.append(context)

    return unique