"""
chunk_and_embed.py

Task 2: Text Chunking, Embedding, and Vector Store Indexing (Optimized for Local Machine)
- Chunks cleaned complaint narratives
- Embeds each chunk using sentence-transformers/all-MiniLM-L6-v2 in batches
- Stores embeddings and metadata in a ChromaDB vector store in batches
- Persists the vector store in vector_store/
- Prints device info and tips for further speedup
"""
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import torch

# Parameters
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 50  # characters
# Use a smaller, faster model for local CPU embedding
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # much faster than all-MiniLM-L6-v2
VECTOR_STORE_DIR = "../vector_store"
BATCH_SIZE = 256  # Try a larger batch size for local CPU

# Load cleaned data
filtered = pd.read_csv("data/filtered_complaints.csv")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[". ", "\n", " "]
)

chunks = []
metadata = []
for idx, row in tqdm(filtered.iterrows(), total=len(filtered)):
    text = str(row["cleaned_narrative"])
    chunked = splitter.split_text(text)
    for chunk in chunked:
        chunks.append(chunk)
        metadata.append({
            "complaint_id": row["Complaint ID"] if "Complaint ID" in row else idx,
            "product": row["Product"]
        })

# Embedding in batches
model = SentenceTransformer(f"sentence-transformers/{EMBEDDING_MODEL}")
print(f"[INFO] Using embedding model: {EMBEDDING_MODEL}")
print("CUDA available:", torch.cuda.is_available())
print("Device:", model.device)
if not torch.cuda.is_available():
    print("\n[INFO] You are running on CPU. For large datasets, embedding will be slow.\n"
          "Tips: This script now uses a smaller, faster model (paraphrase-MiniLM-L3-v2). For even more speed, use a cloud GPU.")

all_embeddings = []
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
    batch_chunks = chunks[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch_chunks, show_progress_bar=False)
    all_embeddings.extend(batch_embeddings)

# Vector store (ChromaDB) - batch insertion
client = chromadb.Client(Settings(
    persist_directory=VECTOR_STORE_DIR
))
collection = client.get_or_create_collection("complaints")

for i in tqdm(range(0, len(all_embeddings), BATCH_SIZE), desc="Vector store batches"):
    batch_embeddings = all_embeddings[i:i+BATCH_SIZE]
    batch_chunks = chunks[i:i+BATCH_SIZE]
    batch_metadata = metadata[i:i+BATCH_SIZE]
    batch_ids = [str(j) for j in range(i, min(i+BATCH_SIZE, len(all_embeddings)))]
    collection.add(
        embeddings=[emb.tolist() for emb in batch_embeddings],
        documents=batch_chunks,
        metadatas=batch_metadata,
        ids=batch_ids
    )



print(f"Vector store saved to {VECTOR_STORE_DIR}")
