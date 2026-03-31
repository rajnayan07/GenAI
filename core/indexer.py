"""
FAISS index builder and manager.
Uses sentence-transformers for LOCAL embeddings (no API calls needed).
Stores vectors in a FAISS index for fast similarity search.
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"
DOCS_PATH = DATA_DIR / "gitlab_docs.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


_embedding_model = None


def get_embedding_model():
    """Lazy-load the sentence-transformers model (cached in memory)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def load_documents() -> list[dict]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DOCS_PATH}. "
            "Run 'python scripts/scrape_gitlab.py' to generate it."
        )
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts locally using sentence-transformers. No API calls needed."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query locally."""
    model = get_embedding_model()
    embedding = model.encode([text], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index using inner product (cosine similarity on normalized vectors)."""
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, chunks: list[dict]):
    import faiss
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index saved: {index.ntotal} vectors, {len(chunks)} chunks")


def load_index():
    """Load a pre-built FAISS index and chunks from disk."""
    import faiss
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Index loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


def build_or_load_index():
    """
    Load existing index from disk, or build a new one.
    No API key needed — embeddings are computed locally.
    """
    from utils.text_processing import chunk_documents

    index, chunks = load_index()
    if index is not None and chunks is not None:
        return index, chunks

    logger.info("Building new index from documents...")
    documents = load_documents()
    chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=200)

    if not chunks:
        raise ValueError("No chunks generated from documents.")

    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks locally (no API calls)...")
    embeddings = embed_texts(texts)

    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    return index, chunks
