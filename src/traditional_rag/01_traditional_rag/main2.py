# LangChain RAG Pipeline:
# This file implements the same RAG flow using LangChain's built-in chains.
# create_stuff_documents_chain() handles combining the retrieved documents, creating the prompt, and passing the context to the LLM.
# create_retrieval_chain() connects the retriever with the document chain, so retrieval, context handling, prompt creation, and LLM invocation are handled automatically.
#
# Run:
# uv run python src\traditional_rag\01_traditional_rag\main2.py



from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from pipeline import build_rag_pipeline

llm, prompt, retriever = build_rag_pipeline()

question = input("Ask a question: ")


# Documents → Prompt → LLM
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# Question → Retriever → Documents → Document Chain
rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# Run complete RAG
response = rag_chain.invoke({
    "input": question
})


print("\n========== ANSWER ==========\n")
print(response["answer"])

