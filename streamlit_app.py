import pickle
import os
import chromadb
import streamlit as st
from dotenv import load_dotenv

from src.pipeline import ask_question


# --- Load everything once, when the app starts ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="contracts")

with open("./data/all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

with open("./data/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


# --- Streamlit UI ---
st.title("Contract Q&A")
st.write("Ask a question about the contracts in the database (CUAD dataset).")

with st.form("question_form"):
    question = st.text_input("Your question:")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Searching contracts and generating answer..."):
        answer = ask_question(
            query=question,
            collection=collection,
            bm25=bm25,
            all_chunks=all_chunks,
            api_key=api_key
        )
    st.write(answer)
