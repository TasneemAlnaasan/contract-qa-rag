
import pickle
import os
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import ask_question


# --- Startup: runs once when the server starts ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="contracts")

with open("./data/all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

with open("./data/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


# --- FastAPI app ---
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = ask_question(
        query=request.question,
        collection=collection,
        bm25=bm25,
        all_chunks=all_chunks,
        api_key=api_key
    )
    return {"answer": answer}