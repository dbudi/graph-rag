from semantic_search import semantic_search_with_score as perform_semantic_search
from entity_extraction_query import extract_entities
from graph_query import graph_query as perform_graph_search
from merge_result import merge_results, deduplicate_context
from generate_answer import generate_answer as generate_answer_from_context
from reranking import rerank
import logging
logger = logging.getLogger(__name__)

def retrieval_pipeline(query: str):
    # Step 1: Semantic Search
    semantic_results = perform_semantic_search(query)
    logger.info(f"Semantic search results: {len(semantic_results)}")


    # Step 2: Extract entities for graph search
    entities = extract_entities(query)  # deduplicate
    logger.info(f"Extracted entities: {entities}")

    # Step 3: Graph Search
    graph_results = perform_graph_search(entities)
    logger.info(f"Graph search results: {len(graph_results)}")

    # Step 4: Merge Results
    merged_results = merge_results(semantic_results, graph_results)
    deduplicate_results = deduplicate_context(merged_results)
    logger.info(f"Deduplicated results: {len(deduplicate_results)}")

    # Step 5: Rerank
    reranked = rerank(query, deduplicate_results)
    logger.info(f"Reranked results: {len(reranked)}")

    # Final context
    top_k = 5
    final_context = reranked[:top_k]   
    logger.info(f"Final context: {len(final_context)}")

    # Step 6: Generate answer
    answer = generate_answer_from_context(query, final_context)
    logger.info(f"Generated answer: {answer}")

    return final_context


# --- Example usage ---
if __name__ == "__main__":
    query = """
    Which universities did Tim Cook graduate from, and in which years? 
    How long Tim Cook has been CEO of Apple? 
    Is Tim Cook have a Cat who's name is Kitty?
    """
    retrieval_pipeline(query)