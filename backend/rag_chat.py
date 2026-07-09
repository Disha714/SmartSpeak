# rag_chat.py
# Retrieval-augmented chat: instead of just stuffing raw conversation
# history into the prompt (which is what the old /llm route did -- it
# concatenated every past Q&A turn as plain text, growing unbounded and
# adding no retrieval signal), this embeds each past turn, indexes it in
# FAISS, and retrieves only the turns most relevant to the *current*
# question before building the prompt.
#
# This is intentionally a small, in-memory, per-process index -- enough to
# demonstrate real RAG mechanics (embed -> index -> retrieve -> augment
# prompt) without needing an external vector DB for a portfolio project.
# Swapping this for a persistent store (Chroma, Pinecone, pgvector) later is
# a drop-in change to _index/_texts.

import numpy as np

_embedder = None
_index = None
_texts = []  # parallel list: _texts[i] corresponds to vector at row i


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_index(dim: int):
    global _index
    if _index is None:
        import faiss
        _index = faiss.IndexFlatL2(dim)
    return _index


def add_turn(question: str, answer: str) -> None:
    """Embed a completed Q&A turn and add it to the retrieval index."""
    embedder = _get_embedder()
    combined = f"Q: {question}\nA: {answer}"
    vector = embedder.encode([combined])
    index = _get_index(vector.shape[1])
    index.add(np.array(vector, dtype="float32"))
    _texts.append(combined)


def retrieve_context(query: str, k: int = 3) -> list:
    """Return up to k most relevant past turns for the given query.
    Returns [] if the index is empty or too small -- callers should treat
    that as "no extra context available", not an error."""
    if not _texts:
        return []
    embedder = _get_embedder()
    vector = embedder.encode([query])
    index = _get_index(vector.shape[1])
    k = min(k, len(_texts))
    _, indices = index.search(np.array(vector, dtype="float32"), k)
    return [_texts[i] for i in indices[0] if 0 <= i < len(_texts)]


def build_messages(query: str, recent_history: list) -> list:
    """Build a chat-completions-style messages list: a system message
    carrying retrieved context, a few recent turns for coherence, then the
    new question. Groq's API (like OpenAI's) takes a messages array, not a
    single raw prompt string, so this replaces the old single-string
    prompt-building approach used with Together's completions endpoint.
    """
    retrieved = retrieve_context(query)

    system_content = "You are a helpful assistant for the SmartSpeak app."
    if retrieved:
        system_content += "\n\nRelevant earlier context from this conversation:\n" + "\n\n".join(
            retrieved
        )

    messages = [{"role": "system", "content": system_content}]

    for entry in recent_history[-4:]:
        q = entry.get("question") if isinstance(entry, dict) else entry[0]
        a = entry.get("answer") if isinstance(entry, dict) else entry[1]
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": query})
    return messages
