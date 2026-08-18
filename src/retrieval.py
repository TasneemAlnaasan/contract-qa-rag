import numpy as np


def semantic_search(query, collection, n_results=10):
    """
    Perform semantic search on the ChromaDB collection using the given query.
    Returns the raw ChromaDB query results (documents, metadatas, ids, distances).
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def bm25_search(query, bm25, all_chunks, n_results=10):
    """
    Perform BM25 keyword search over all_chunks using the given query.
    Returns the top n_results chunks (full dicts: text, source, chunk_id),
    ranked by BM25 score.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_n_indices = np.argsort(scores)[::-1][:n_results]

    results = [all_chunks[i] for i in top_n_indices]
    return results


def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    """
    Combine semantic search and BM25 search results using
    Reciprocal Rank Fusion (RRF). Returns a dict of {chunk_id: rrf_score}.
    """
    scores = {}

    semantic_ids = semantic_results["ids"][0]
    for rank, chunk_id in enumerate(semantic_ids):
        rrf_score = 1 / (k + rank)
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["chunk_id"]
        rrf_score = 1 / (k + rank)
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    return scores


def get_chunks_by_ids(chunk_ids, all_chunks):
    """
    Given a list of chunk_ids, return the full chunk dicts (text, source, chunk_id)
    from all_chunks. Builds a lookup dictionary first for fast access.
    """
    chunks_lookup = {chunk["chunk_id"]: chunk for chunk in all_chunks}

    result = [chunks_lookup[cid] for cid in chunk_ids if cid in chunks_lookup]

    return result
