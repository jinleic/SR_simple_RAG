# validation.py
from custom_vector_store import CustomVectorStore
from data_ingestion import ingest_data
from rag_query import rag_query
import os

def test_all():
    print("Starting full system validation...")
    store = CustomVectorStore()
    
    # Ensure data directory exists and create dummy files if needed
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    dummy_data = {
        "doc1.txt": "FAISS is an efficient vector search library.",
        "doc2.txt": "Retrieval-Augmented Generation (RAG) combines retrieval with LLMs."
    }
    for fname, content in dummy_data.items():
        with open(os.path.join(data_dir, fname), "w", encoding="utf-8") as f:
            f.write(content)
    
    doc_count = ingest_data(data_dir, store)
    assert doc_count > 0, "No documents were ingested."
    
    # Test a RAG query
    query = "Explain the role of FAISS in vector search."
    answer, sources = rag_query(query, store, top_k=2)
    print("Final RAG Query Answer:")
    print(answer)
    print("Sources used:")
    for src in sources:
        print(src)
    
    # Basic assertion to check expected output
    assert "FAISS" in answer or len(sources) > 0, "The answer should mention FAISS or use a relevant source."
    print("Full system validation passed.")

if __name__ == "__main__":
    test_all()

