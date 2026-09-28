from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from reranking.rerank_documents import rerank_documents


class RerankingRetriever(BaseRetriever):

    retriever: object
    reranker: object
    top_k: int

    def _get_relevant_documents(self, query: str) -> list[Document]:

        # Step 1: Retrieve the initial candidate documents.
        #
        # The base retriever performs the first-stage retrieval,
        # usually using vector similarity, and returns a larger
        # set of potentially relevant documents.
        documents = self.retriever.invoke(query)

        # Step 2: Re-rank the retrieved documents.
        #
        # The re-ranker compares the query with each candidate
        # document and assigns a relevance score.
        #
        # The result contains both the Document and its score:
        # {
        #     "document": Document(...),
        #     "score": relevance_score
        # }
        reranked_results = rerank_documents(
            query=query,
            documents=documents,
            reranker=self.reranker,
            top_k=self.top_k
        )

        # Step 3: Return only Document objects.
        #
        # rerank_documents() returns dictionaries containing both
        # the Document and its relevance score.
        # rerank_documents() returns dictionaries containing both:
        #   - document → the actual Document object
        #   - score    → the relevance score calculated by the re-ranker
        #
        #
        # Therefore, extract the Document from each result before
        # passing the documents to create_retrieval_chain().
        # The score is useful for debugging and evaluating the re-ranking,
        # but create_retrieval_chain() expects the retriever to return
        # a list of Document objects. Therefore retriever must return
        # a list of Document objects.
        #
        # Therefore, we extract only the Document from each re-ranked result
        # before passing them to the LangChain retrieval chain.
        return [
            item["document"]
            for item in reranked_results
        ]