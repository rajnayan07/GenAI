"""
RAG retriever: embeds user queries locally and searches the FAISS index
to find the most relevant document chunks.
"""

import logging

import numpy as np

from core.indexer import embed_query

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.
    Embeddings are computed locally — no API key needed.
    """
    import faiss

    query_embedding = embed_query(query)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if score < score_threshold:
            continue

        chunk = chunks[idx].copy()
        chunk["relevance_score"] = float(score)
        results.append(chunk)

    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    if results:
        logger.info(
            f"Query: '{query[:60]}...' -> {len(results)} results "
            f"(top score: {results[0]['relevance_score']:.3f})"
        )
    else:
        logger.info(f"Query: '{query[:60]}...' -> 0 results")

    return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    if not results:
        return "No relevant information found in the GitLab handbook."

    context_parts = []
    seen_titles = set()

    for r in results:
        source_label = f"[{r['title']}]({r['url']})" if r.get("url") else r.get("title", "Unknown")

        if r["title"] not in seen_titles:
            context_parts.append(
                f"**Source: {source_label}** (Section: {r.get('section', 'General')}, "
                f"Relevance: {r['relevance_score']:.0%})\n{r['text']}"
            )
            seen_titles.add(r["title"])
        else:
            context_parts.append(
                f"**Source: {source_label}** (continued)\n{r['text']}"
            )

    return "\n\n---\n\n".join(context_parts)


def get_source_citations(results: list[dict]) -> list[dict]:
    """Extract unique source citations from results for display in the UI."""
    seen = set()
    citations = []

    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            citations.append({
                "title": r.get("title", "Unknown"),
                "url": url,
                "section": r.get("section", "General"),
                "relevance": r.get("relevance_score", 0),
            })

    return citations
