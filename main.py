# main.py
import argparse
from custom_vector_store import CustomVectorStore
from data_ingestion import ingest_data
from rag_query import rag_query
import os

def main():
    parser = argparse.ArgumentParser(description="Run a Retrieval-Augmented Generation (RAG) pipeline.")
    parser.add_argument("--query", type=str, help="The query to ask the RAG system.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing text documents.")
    args = parser.parse_args()
    
    # Initialize vector store and ingest data
    store = CustomVectorStore()
    if not os.path.isdir(args.data_dir):
        print(f"Data directory {args.data_dir} not found. Exiting.")
        return
    doc_count = ingest_data(args.data_dir, store)
    if doc_count == 0:
        print("No documents ingested. Please add text files to the data directory.")
        return
    
    # If a query is provided, run the RAG query workflow
    if args.query:
        answer, sources = rag_query(args.query, store, top_k=2)
        print("\n--- RAG Query Result ---")
        print("Answer:")
        print(answer)
        print("\nSource Documents:")
        for i, src in enumerate(sources):
            print(f"Source {i+1}: {src}")
    else:
        print("No query provided. Use --query to ask a question.")

if __name__ == "__main__":
    main()

