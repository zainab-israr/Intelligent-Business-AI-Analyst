"""
Query the RAG vector store to retrieve relevant documents.
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and doc metadata
index = faiss.read_index("rag/faiss_index.bin")
with open("rag/doc_names.pkl", "rb") as f:
    doc_names = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_rag(question, top_k=2):
    """
    Query the vector store with a question and return top_k relevant docs.
    """
    q_emb = model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "doc_name": doc_names[idx],
            "distance": float(distances[0][i])
        })
    return results

# Example usage
if __name__ == "__main__":
    question = "What are the sales trends in the North region?"
    results = query_rag(question)
    for r in results:
        print(f"Document: {r['doc_name']}, Distance: {r['distance']:.4f}")
