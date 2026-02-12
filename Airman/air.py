from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import requests

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("data/vector_store/index.faiss")

# Load metadata
with open("data/vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("✅ AIR system loaded")
app = FastAPI(title="AIRMAN Aviation Document AI Chat")
templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (safe for assignment)
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

def retrieve_chunks(question, top_k=8):
    q_embedding = model.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx].copy()
        chunk["score"] = float(distances[0][i])
        results.append(chunk)

    # 🔹 Boost chunks that directly contain the main keyword
    question_lower = question.lower()

    for r in results:
        if question_lower in r["text"].lower():
            r["score"] *= 0.8  # slight boost

    # Sort again after boosting
    results = sorted(results, key=lambda x: x["score"])

    return results


def is_relevant(results, threshold=1.2):
    if not results:
        return False
    return results[0]["score"] < threshold

def clean_text(text):
    # Remove weird OCR artifacts
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ascii
    text = re.sub(r"\b[a-zA-Z0-9]{1,3}\b", " ", text)  # remove tiny broken words
    text = re.sub(r"\s+", " ", text)  # normalize spaces

    sentences = text.split(".")
    cleaned = []

    for s in sentences:
        s = s.strip()

        # Keep meaningful sentences only
        if len(s) > 40 and not re.search(r"[0-9]{3,}", s):
            cleaned.append(s)

    return ". ".join(cleaned[:2]) + "." if cleaned else ""




def generate_answer(results, question):
    if not results:
        return "", []

    best_chunk = results[0]["text"]
    source = results[0]["source"]

    prompt = f"""
You are an aviation document assistant.

Answer the question ONLY using the provided context.
If the answer is not found in the context, say:
"This information is not found in the provided aviation documents."

Context:
{best_chunk}

Question:
{question}

Provide a clear and professional answer.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    return answer.strip(), [source]



def ask_air(question):
    results = retrieve_chunks(question)

    if not is_relevant(results):
        return {
            "answer": "This information is not found in the provided aviation documents.",
            "sources": []
        }

    answer, sources = generate_answer(results, question)


    return {
        "answer": answer,
        "sources": sources
    }

"""
if __name__ == "__main__":
    while True:
        q = input("\nAsk AIR (type 'exit' to quit): ")
        if q.lower() == "exit":
            break

        response = ask_air(q)
        print("\nANSWER:\n", response["answer"])
        print("\nSOURCES:", response["sources"])
"""
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
def ask_api(q: Question):
    return ask_air(q.question)


