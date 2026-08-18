from src.query_rewriting import rewrite_query
from src.retrieval import semantic_search, bm25_search, reciprocal_rank_fusion, get_chunks_by_ids
from src.reranking import rerank
from src.generation import generate_answer


def ask_question(query, collection, bm25, all_chunks, api_key, n_results=10, top_n=5):
    """
    Full end-to-end RAG pipeline: rewrite -> hybrid retrieval -> fusion
    -> rerank -> generate. Takes the user's original query and returns
    the final answer (in the same language as the original query).
    """
    # Step 1: rewrite the query (clearer, in English, for better retrieval)
    rewritten = rewrite_query(query, api_key)

    # Step 2: search using both methods
    semantic_results = semantic_search(rewritten, collection, n_results=n_results)
    bm25_results = bm25_search(rewritten, bm25, all_chunks, n_results=n_results)

    # Step 3: combine results using RRF
    fused_scores = reciprocal_rank_fusion(semantic_results, bm25_results)

    # Step 4: take the top N ids, then fetch their full chunk data
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    top_chunk_ids = [chunk_id for chunk_id, score in sorted_fused[:n_results]]
    candidates = get_chunks_by_ids(top_chunk_ids, all_chunks)

    # Step 5: re-rank the candidates against the rewritten (English) query
    reranked = rerank(rewritten, candidates, top_n=top_n)

    # Step 6: generate the final answer using the original query (same language)
    answer = generate_answer(query, reranked, api_key)

    return answer
