"""
Text processing utilities for chunking and cleaning documents.
"""

import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks by sentence boundaries.
    Tries to keep chunks semantically coherent by splitting on
    paragraph and sentence boundaries rather than mid-word.
    """
    text = clean_text(text)

    if len(text) <= chunk_size:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+|\n\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > chunk_overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s)

            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split documents into chunks, preserving metadata for source attribution.
    Each chunk includes the title, url, section, and chunk index.
    """
    all_chunks = []

    for doc in documents:
        content = doc.get("content", "")
        text_chunks = chunk_text(content, chunk_size, chunk_overlap)

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "section": doc.get("section", ""),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            })

    return all_chunks
