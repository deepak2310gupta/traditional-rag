from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(embedding_model: str):

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    return embeddings


# Quick Note:

# Embedding Your Chunks

# embedding_model = get_embedding_model()

# vectors = embedding_model.embed_documents(
#     [chunk.page_content for chunk in chunks]
# )

# print(len(vectors))


# Better Approach

# Don't generate vectors yourself.
# Most vector databases generate embeddings automatically.

# Example:
# vector_store = FAISS.from_documents(
#     chunks,
#     embedding_model
# )

# LangChain will: 
# Read chunk text
# Create embeddings 
# Store vectors
# automatically.