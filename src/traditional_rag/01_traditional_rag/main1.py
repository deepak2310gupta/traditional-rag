# Manual RAG Pipeline:
# This file implements the RAG flow step-by-step manually.
# We explicitly call the retriever, create the context, build the prompt, and invoke the LLM.
# This is mainly useful for understanding how RAG works internally.
#
# Run:
# uv run python src\traditional_rag\01_traditional_rag\main1.py



from pipeline import build_rag_pipeline


llm, prompt, retriever = build_rag_pipeline()

question = input("Ask a question: ")


# Retrieve relevant documents
retrieved_documents = retriever.invoke(question)
print(f"Retrieved documents: {len(retrieved_documents)}")


# Print retrieved document details
for i, document in enumerate(retrieved_documents, start=1):
    print(f"\n    Document {i}:")
    print(f"      Source: {document.metadata.get('source')}")
    print(f"      Page  : {document.metadata.get('page')}")
    print(f"      Size  : {len(document.page_content)} characters")
    print(f"      Content : {document.page_content[:300].replace(chr(10), ' ')}")

# Create context
context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
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