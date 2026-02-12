import os
import re
import json

INPUT_DIR = "data/extracted_text"
OUTPUT_FILE = "data/cleaned_chunks.json"

CHUNK_SIZE = 200   # words
OVERLAP = 40      # words


def clean_text(text):
    # Remove page markers
    text = re.sub(r"--- Page \d+ ---", "", text)

    # Remove non-ASCII characters (OCR junk)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove weird symbol clusters
    text = re.sub(r"[^\w\s.,;:()-]", " ", text)

    # Remove words that look like OCR noise
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)

    # Remove repeated headers like "The Atmosphere"
    text = re.sub(r"(The Atmosphere\s*){2,}", "The Atmosphere ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def run():
    all_chunks = []
    chunk_id = 0

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        print(f"Processing: {filename}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        for chunk in chunks:
            if len(chunk.split()) < 50:
                continue

            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk,
                "source": filename
            })
            chunk_id += 1

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n✅ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
