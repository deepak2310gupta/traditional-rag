# LangChain RAG + Re-ranking Pipeline:
# This file implements the RAG + re-ranking flow using LangChain's built-in chains.
# RerankingRetriever handles document retrieval and re-ranking, while
# create_stuff_documents_chain() handles context creation, prompt formatting, and LLM invocation.
# create_retrieval_chain() connects the retriever and document chain into a complete RAG pipeline.
#
# Run:
# uv run python src\traditional_rag\02_traditional_rerank_rag\main2.py


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


from reranking.reranker import get_reranker
from reranking.reranking_retriever import RerankingRetriever
from pipeline import build_rag_pipeline

from config import RERANK_TOP_K


llm, prompt, retriever = build_rag_pipeline()

question = input("Ask a question: ")

# Create the re-ranker model.
reranker = get_reranker()

# Wrap the original retriever with the re-ranker
#
# Flow:
# Question
#    ↓
# Base Retriever → retrieves initial documents
#    ↓
# Re-ranker → reorders documents based on relevance
#    ↓
# top_k documents → passed to the LLM
reranking_retriever = RerankingRetriever(
    retriever=retriever, 
    reranker=reranker,
    top_k=RERANK_TOP_K
)

# Documents → Prompt → LLM
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# Question → Retriever → Documents → Document Chain
rag_chain = create_retrieval_chain(
    reranking_retriever, # We're simply replacing: normal retriever with: reranking retriever
    document_chain
)

# Run complete RAG
response = rag_chain.invoke({
    "input": question
})


print("\n========== ANSWER ==========\n")
print(response["answer"])



