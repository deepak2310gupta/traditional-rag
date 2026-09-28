from traditional_rag.common.ingestion.loader import load_documents
from traditional_rag.common.chunking.splitter import create_document_chunks
from traditional_rag.common.embeddings.embedding_model import get_embedding_model
from traditional_rag.common.vector_store.vector_store import create_vector_store
from traditional_rag.common.retrieval.retriever import create_retriever
from traditional_rag.common.generation.llm import create_llm
from traditional_rag.common.generation.prompt import create_prompt

from config import DIRECTORY_PATH
from config import CHUNK_SIZE, CHUNK_OVERLAP
from config import EMBEDDING_MODEL
from config import TOP_K


def build_rag_pipeline():

    # 1. Load documents
    documents = load_documents(DIRECTORY_PATH)
    print(f"Loaded {len(documents)} documents")
    
    # 2. Split documents
    chunks = create_document_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")
    print(f"Created chunks: {len(chunks)}")

    # 3. Create embedding model
    embeddings = get_embedding_model(EMBEDDING_MODEL)

    # 4. Create vector store
    vector_store = create_vector_store(chunks,embeddings)
    print("Vector store created successfully")

    # 5. Create retriever
    retriever = create_retriever(vector_store, TOP_K)

    # 6. Create LLM
    llm = create_llm()
    
    # 7. Create Prompt
    prompt = create_prompt()

    return llm, prompt, retriever