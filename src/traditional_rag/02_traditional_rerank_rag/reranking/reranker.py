from sentence_transformers import CrossEncoder


def get_reranker():

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return reranker