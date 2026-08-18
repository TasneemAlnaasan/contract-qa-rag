
from sentence_transformers import CrossEncoder

def rerank(query, candidates, top_n=5):
    """
    Re-rank a list of candidate chunks against the query using a
    cross-encoder model. Returns the top_n chunks (full dicts),
    ordered by relevance score (highest first).
    """
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs)

    candidate_score_pairs = list(zip(candidates, scores))
    sorted_pairs = sorted(candidate_score_pairs, key=lambda x: x[1], reverse=True)

    top_candidates = [candidate for candidate, score in sorted_pairs[:top_n]]

    return top_candidates
