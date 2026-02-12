import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer 

CHUNKS_FILE = "data/cleaned_chunks.json"
VECTOR_DIR = "data/vector_store"

MODEL_NAME = "all-MiniLM-L6-v2"  # fast & good


def run():
    print("🔹 Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print("🔹 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("🔹 Creating embeddings (this may take a minute)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    print(f"🔹 Embedding dimension: {dim}")

    print("🔹 Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

    with open(os.path.join(VECTOR_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("✅ FAISS index and metadata saved successfully")


if __name__ == "__main__":
    run()