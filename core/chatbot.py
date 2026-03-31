"""
Groq-powered chatbot with RAG (Retrieval Augmented Generation).
Uses Llama 3.3 via Groq's free API for fast, high-quality responses
about GitLab's Handbook and Direction pages.
"""

import logging

from groq import Groq

from core.guardrails import build_system_guardrail

logger = logging.getLogger(__name__)

GENERATION_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = f"""You are **GitLab Handbook Assistant**, an AI helper that answers questions
about GitLab's public Handbook and Direction pages.

Your role:
- Help employees and aspiring employees learn about GitLab's culture, values,
  processes, product direction, engineering practices, and company policies.
- Provide accurate, well-structured answers based on the provided context.
- Cite sources when possible by referencing the handbook page title.
- Be friendly, professional, and encouraging.

{build_system_guardrail()}

When answering:
1. Ground your answers in the provided context from the GitLab handbook.
2. If the context doesn't fully cover the question, say what you know and
   note what information might be missing.
3. Use bullet points and headers for readability when appropriate.
4. If asked about something not covered in the context, be honest about
   the limitation and suggest where to look.
5. Encourage users to check the official handbook for the most up-to-date info.
"""


def create_chat_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def build_rag_prompt(query: str, context: str, relevance_hint: str = "") -> str:
    """Build the RAG prompt combining user query with retrieved context."""
    hint_section = f"\n**Note:** {relevance_hint}\n" if relevance_hint else ""

    return f"""Based on the following context from GitLab's Handbook and Direction pages,
answer the user's question. If the context doesn't contain enough information,
say so honestly.
{hint_section}
--- CONTEXT START ---
{context}
--- CONTEXT END ---

**User Question:** {query}

Provide a clear, helpful answer. Cite specific handbook pages when referencing information."""


def _build_messages(chat_history: list[dict], prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_response(
    client: Groq,
    chat_history: list[dict],
    query: str,
    context: str,
    relevance_hint: str = "",
) -> str:
    """Generate a response using Groq with RAG context."""
    prompt = build_rag_prompt(query, context, relevance_hint)
    messages = _build_messages(chat_history, prompt)

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=2048,
            top_p=0.9,
        )
        return response.choices[0].message.content or (
            "I wasn't able to generate a response. Please try rephrasing your question."
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return (
            f"I encountered an error while generating a response: {str(e)}. "
            "Please try again or rephrase your question."
        )


def generate_response_stream(
    client: Groq,
    chat_history: list[dict],
    query: str,
    context: str,
    relevance_hint: str = "",
):
    """Generate a streaming response using Groq with RAG context."""
    prompt = build_rag_prompt(query, context, relevance_hint)
    messages = _build_messages(chat_history, prompt)

    try:
        stream = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=2048,
            top_p=0.9,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
    except Exception as e:
        logger.error(f"Stream generation failed: {e}")
        yield (
            f"I encountered an error: {str(e)}. "
            "Please try again or rephrase your question."
        )
