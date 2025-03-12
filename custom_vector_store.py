# custom_vector_store.py
import numpy as np
import faiss
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class CustomVectorStore:
    def __init__(self, embedding_dim=384):
        # Initialize FAISS index (Flat index for simplicity)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.documents = []  # store original texts

    def add_documents(self, docs):
        """Embed and add a list of documents."""
        if not docs:
            return
        # Get embeddings for all docs
        vectors = self.embeddings.embed_documents(docs)
        vectors = np.array(vectors).astype("float32")
        # Add to FAISS index and store docs
        self.index.add(vectors)
        self.documents.extend(docs)

    def query(self, query_text, top_k=2):
        """Query the index with a text string."""
        query_vec = np.array(self.embeddings.embed_query(query_text)).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        results = [(self.documents[idx], float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx < len(self.documents)]
        return results

# ----------------- Test Function ----------------- #
def test_custom_vector_store():
    store = CustomVectorStore()
    sample_docs = ["Hello world!", "RAG is cool"]
    store.add_documents(sample_docs)
    results = store.query("Tell me about RAG", top_k=2)
    print("Vector Store Query Results:")
    for doc, dist in results:
        print(f"Document: {doc} | Distance: {dist}")

if __name__ == "__main__":
    test_custom_vector_store()

