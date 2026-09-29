# Manual RAG + Re-ranking Pipeline:
# This file implements the RAG + re-ranking flow step-by-step manually.
# We explicitly call the retriever, re-rank the retrieved documents, create the context,
# build the prompt, and invoke the LLM.
# This is mainly useful for understanding how re-ranking fits into the RAG pipeline.
#
# Run:
# uv run python src\traditional_rag\02_traditional_rerank_rag\main1.py


from pipeline import build_rag_pipeline

from reranking.reranker import get_reranker
from reranking.rerank_documents import rerank_documents

from config import RERANK_TOP_K


llm, prompt, retriever = build_rag_pipeline()

question = input("Ask a question: ")

# Retrieve relevant documents
retrieved_documents = retriever.invoke(question)
print(f"Retrieved documents: {len(retrieved_documents)}")


print("\n" + "=" * 80)
print("BEFORE RERANKING")
print("=" * 80)

# Print retrieved document details
for i, document in enumerate(retrieved_documents, start=1):
    print(f"\n    Document {i}:")
    print(f"      Source: {document.metadata.get('source')}")
    print(f"      Page  : {document.metadata.get('page')}")
    print(f"      Size  : {len(document.page_content)} characters")
    print(f"      Content : {document.page_content[:300].replace(chr(10), ' ')}")


reranker = get_reranker()

# Rerank candidates
reranked_documents = rerank_documents(
    query=question,
    documents=retrieved_documents,
    reranker=reranker,
    top_k=RERANK_TOP_K
)


print("\n" + "=" * 80)
print("AFTER RERANKING")
print("=" * 80)


for i, result in enumerate(reranked_documents, start=1):

    doc = result["document"]
    score = result["score"]

    print(f"\n--- Rank {i} | Score: {score:.4f} ---")
    print(f"Source  : {doc.metadata.get('source', 'N/A')}")
    print(f"Page    : {doc.metadata.get('page_label', 'N/A')}")
    print(f"Size  : {len(doc.page_content)} characters")
    print(f"Content : {doc.page_content[:300].replace(chr(10), ' ')}")


# Create context
context = "\n\n".join(
    document.page_content
    for result in reranked_documents
    for document in [result["document"]]
)
print(f"Context size: {len(context)} characters")

# Create prompt
formatted_prompt = prompt.invoke({
    "context": context,
    "input": question
})

# Generate answer
answer = llm.invoke(formatted_prompt)

print("\n========== ANSWER ==========\n")
print(answer)

