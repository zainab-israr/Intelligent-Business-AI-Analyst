"""
Build FAISS vector store from business documents for RAG queries.
"""

import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# Ensure docs folder exists
DOCS_DIR = "docs"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    # Add sample docs
    with open(os.path.join(DOCS_DIR, "monthly_report.txt"), "w") as f:
        f.write("This is the monthly report. Sales increased by 15% in the North region.")
    with open(os.path.join(DOCS_DIR, "hr_policy.txt"), "w") as f:
        f.write("All employees must follow company guidelines for work hours and benefits.")

# Load all docs
docs = []
doc_names = []
for file in os.listdir(DOCS_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DOCS_DIR, file), "r") as f:
            docs.append(f.read())
            doc_names.append(file)

# Initialize SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# Compute embeddings
embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and metadata
os.makedirs("rag", exist_ok=True)
faiss.write_index(index, "rag/faiss_index.bin")
with open("rag/doc_names.pkl", "wb") as f:
    pickle.dump(doc_names, f)

print("RAG vector store built and saved in 'rag/'")
