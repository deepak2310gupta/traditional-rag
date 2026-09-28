from langchain_core.documents import Document

def rerank_documents(
    query: str,
    documents: list[Document],
    reranker,
    top_k: int
):

    # Create query-document pairs
    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    scored_documents = list(
        zip(documents, scores)
    )
    
    # Sort by score: highest first
    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k documents with scores
    reranked_documents = [
        {
            "document": document,
            "score": float(score),
        }
        for document, score in scored_documents[:top_k]
    ]

    return reranked_documents