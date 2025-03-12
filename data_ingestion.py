# data_ingestion.py
import os
import glob
from custom_vector_store import CustomVectorStore

def preprocess_text(text):
    # Simple preprocessing: strip whitespace and remove duplicate spaces
    return " ".join(text.strip().split())

def ingest_data(data_dir, vector_store):
    file_pattern = os.path.join(data_dir, "*.txt")
    files = glob.glob(file_pattern)
    docs = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = preprocess_text(f.read())
            docs.append(content)
    # Deduplicate documents
    docs = list(set(docs))
    vector_store.add_documents(docs)
    return len(docs)

# ----------------- Test Function ----------------- #
def test_ingest_data():
    store = CustomVectorStore()
    # Create dummy data if folder is empty
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    sample_files = {
        "sample1.txt": "Document about FAISS and vector search.",
        "sample2.txt": "RAG systems combine retrieval with generation."
    }
    for filename, content in sample_files.items():
        with open(os.path.join(data_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
    doc_count = ingest_data(data_dir, store)
    print(f"Ingested {doc_count} documents.")
    # Query the store to validate ingestion
    results = store.query("FAISS", top_k=1)
    assert len(results) > 0, "No documents found in vector store after ingestion."
    print("Data ingestion test passed.")

if __name__ == "__main__":
    test_ingest_data()

