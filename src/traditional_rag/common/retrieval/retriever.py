from langchain_core.vectorstores import VectorStoreRetriever

def create_retriever(vector_store, top_k) -> VectorStoreRetriever:

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k
        }
    )

    return retriever




# Better Retrieval Strategy: MMR
# For RAG, I usually prefer MMR over plain similarity because it reduces duplicate chunks.