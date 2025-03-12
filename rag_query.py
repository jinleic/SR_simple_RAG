# rag_query.py
from custom_vector_store import CustomVectorStore
from ollama_integration import query_ollama

def format_prompt(query, retrieved_docs):
    """
    Formats the final prompt by combining the user query with retrieved document context.
    """
    context = "\n\n".join([f"Source {i+1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
    final_prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return final_prompt

def rag_query(query, vector_store, top_k=2):
    """
    End-to-end RAG workflow:
      1. Retrieve top-k documents.
      2. Format prompt with retrieved context.
      3. Query the LLM via Ollama.
    Returns the answer and the list of source documents.
    """
    retrieved_docs = vector_store.query(query, top_k=top_k)
    if not retrieved_docs:
        return "No context found.", []
    prompt = format_prompt(query, retrieved_docs)
    answer = query_ollama(prompt, system_prompt="You are a helpful expert.")
    return answer, [doc for doc, _ in retrieved_docs]

# ----------------- Test Function ----------------- #
def test_rag_query():
    store = CustomVectorStore()
    # Add sample documents for testing
    sample_docs = [
        "FAISS is a library for efficient similarity search.",
        "It is widely used in vector search applications."
    ]
    store.add_documents(sample_docs)
    query = "What is FAISS?"
    answer, sources = rag_query(query, store, top_k=2)
    print("RAG Query Answer:")
    print(answer)
    print("Sources:")
    for source in sources:
        print(source)
    assert "FAISS" in answer or len(sources) > 0, "Expected context about FAISS not found."

if __name__ == "__main__":
    test_rag_query()

