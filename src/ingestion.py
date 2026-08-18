
import chromadb
from rank_bm25 import BM25Okapi

def chunk_text(text, chunk_size=400, overlap=80):
    """
    Split text into chunks of a given size (in words), with overlap
    between consecutive chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, keeping overlap

    return chunks


def build_all_chunks(unique_contracts, chunk_size=400, overlap=80):
    """
    Apply chunking to all contracts, and return a single unified list
    of chunks, each with its source (contract title) and a unique chunk_id.
    """
    all_chunks = []

    for title, context in unique_contracts.items():
        contract_chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(contract_chunks):
            chunk_dict = {
                "text": chunk,
                "source": title,
                "chunk_id": f"{title}_{i}"
            }
            all_chunks.append(chunk_dict)

    return all_chunks


def build_chroma_index(all_chunks, persist_path="../chroma_db", collection_name="contracts", batch_size=5000):
    """
    Build a ChromaDB index (persistent, on disk) from a list of chunks.
    ChromaDB automatically generates embeddings for each document.
    Adds data in batches to avoid exceeding ChromaDB's max batch size.
    """
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.create_collection(name=collection_name)

    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in all_chunks]
    ids = [chunk["chunk_id"] for chunk in all_chunks]

    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        print(f"Added batch {i} to {i + batch_size}")

    return collection

def build_bm25_index(all_chunks):
    documents = [chunk["text"] for chunk in all_chunks]
    tokenized_corpus = [text.lower().split() for text in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25