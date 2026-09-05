# Contract QA RAG

An Advanced RAG (Retrieval-Augmented Generation) system for answering natural-language questions about legal contracts, built as part of a GenAI engineering portfolio.

**Live demo:** [(https://appapppy-x97zrtnp7hwxez8fqwkvrt.streamlit.app/)]

## Overview

This project answers questions about legal contracts using a hybrid retrieval pipeline combining semantic search, keyword search, query rewriting, and re-ranking — built step by step to deeply understand each component of an Advanced RAG system, rather than using an out-of-the-box framework.

## Architecture

1. **Query Rewriting** — the user's question is rephrased (and translated to English if needed) using Groq, to improve retrieval accuracy.
2. **Hybrid Retrieval**:
   - **Semantic Search** (ChromaDB + sentence-transformers embeddings) — finds chunks that match the *meaning* of the query.
   - **BM25 Search** (rank_bm25) — finds chunks that match *exact keywords/terms*.
   - **Reciprocal Rank Fusion (RRF)** — combines both result sets into a single ranked list.
3. **Re-ranking** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the top candidates against the query for higher precision.
4. **Generation** — Groq generates the final answer, grounded only in the retrieved contract excerpts (in the user's original language).

## Dataset

[CUAD (Contract Understanding Atticus Dataset)](https://huggingface.co/datasets/theatticusproject/cuad-qa) — 408 unique legal contracts, chunked into ~10,457 passages (400 words per chunk, 80-word overlap).

## Tech Stack

- **LLM**: Groq (openai/gpt-oss-20b)
- **Vector DB**: ChromaDB (persistent)
- **Keyword Search**: rank_bm25
- **Re-ranking**: sentence-transformers CrossEncoder
- **Backend**: FastAPI (alternative API interface)
- **Frontend**: Streamlit
- **Deployment**: Streamlit Community Cloud

## Project Structure

```
contract-qa-rag/
├── data/                  # Persisted chunks and BM25 index (pickle)
├── notebooks/             # Data exploration and step-by-step development notebook
├── src/
│   ├── ingestion.py       # Chunking, ChromaDB indexing, BM25 indexing
│   ├── query_rewriting.py # Query rewriting via Groq
│   ├── retrieval.py       # Semantic search, BM25 search, RRF fusion
│   ├── reranking.py       # Cross-encoder re-ranking
│   ├── generation.py      # Final answer generation via Groq
│   └── pipeline.py        # End-to-end orchestration (ask_question)
├── app/
│   └── main.py            # FastAPI alternative interface
├── streamlit_app.py        # Streamlit UI (main deployed app)
├── chroma_db/              # Persistent ChromaDB store
├── Dockerfile               # Alternative containerized deployment
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

You'll need a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

## Notes

This is a learning-focused portfolio project: core RAG components (chunking, fusion) were implemented manually rather than via frameworks like LangChain, to build a deeper understanding of how Advanced RAG systems work under the hood.