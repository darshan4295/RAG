import os
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

def main():
    store = FaissVectorStore("faiss_store")
    
    # Option 1: Build from scratch (first time)
    if not os.path.exists(os.path.join("faiss_store", "faiss.index")):
        print("Building index for the first time...")
        docs = load_all_documents("data")
        store.build_from_documents(docs)
    
    # Option 2: Load existing index
    else:
        print("Loading existing index...")
        store.load()
    
    # Query the store
    query = "What is attention mechanism?"
    results = store.query(query, top_k=3)
    
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text: {result['metadata']['text'][:300]}...")
        print()

if __name__ == "__main__":
    main()