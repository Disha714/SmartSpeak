# rag_chat.py
# Retrieval-augmented chat: instead of just stuffing raw conversation
# history into the prompt (which is what the old /llm route did -- it
# concatenated every past Q&A turn as plain text, growing unbounded and
# adding no retrieval signal), this embeds each past turn, indexes it in
# FAISS, and retrieves only the turns most relevant to the *current*
# question before building the prompt.
#
# Two upgrades over a minimal RAG demo:
#
# 1. PERSISTENCE. The index used to live only in process memory -- every
#    server restart silently wiped all retrieval history. It's now saved to
#    disk (rag_store/) after every write and reloaded on first use, so
#    retrieval survives restarts like a real deployment would need.
#
# 2. HYBRID RETRIEVAL. Dense embedding search (FAISS) is good at semantic
#    paraphrase ("what did I say about my trip" matching "the Goa vacation")
#    but weak on exact tokens it wasn't trained to weight heavily -- specific
#    names, IDs, numbers. Sparse keyword search (BM25) is the opposite: exact
#    on tokens, blind to paraphrase. This combines both result rankings with
#    Reciprocal Rank Fusion (RRF) -- a standard, simple hybrid-search
#    technique (used in e.g. Elasticsearch's hybrid retriever) that doesn't
#    require normalizing two different scoring scales, just combining ranks.

import json
import os

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_store")
_INDEX_PATH = os.path.join(_DATA_DIR, "faiss.index")
_TEXTS_PATH = os.path.join(_DATA_DIR, "texts.json")

_embedder = None
_index = None
_texts = []  # parallel list: _texts[i] corresponds to vector at row i
_loaded_from_disk = False


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _ensure_loaded():
    """Load a previously persisted index + texts from disk, once, on first
    real use. Safe to call repeatedly -- a no-op after the first call."""
    global _index, _texts, _loaded_from_disk
    if _loaded_from_disk:
        return
    _loaded_from_disk = True

    if os.path.exists(_INDEX_PATH) and os.path.exists(_TEXTS_PATH):
        import faiss
        _index = faiss.read_index(_INDEX_PATH)
        with open(_TEXTS_PATH, "r", encoding="utf-8") as f:
            _texts = json.load(f)


def _get_index(dim: int):
    global _index
    _ensure_loaded()
    if _index is None:
        import faiss
        _index = faiss.IndexFlatL2(dim)
    return _index


def _persist() -> None:
    """Write the current index + texts to disk so retrieval history
    survives a server restart."""
    import faiss
    os.makedirs(_DATA_DIR, exist_ok=True)
    faiss.write_index(_index, _INDEX_PATH)
    with open(_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_texts, f)


def add_turn(question: str, answer: str) -> None:
    """Embed a completed Q&A turn, add it to the retrieval index, and
    persist immediately so it survives a restart."""
    _ensure_loaded()
    embedder = _get_embedder()
    combined = f"Q: {question}\nA: {answer}"
    vector = embedder.encode([combined])
    index = _get_index(vector.shape[1])
    index.add(np.array(vector, dtype="float32"))
    _texts.append(combined)
    _persist()


def _dense_rank(query: str, k: int) -> list:
    """Return indices into _texts, best dense (embedding) match first."""
    if not _texts:
        return []
    embedder = _get_embedder()
    vector = embedder.encode([query])
    index = _get_index(vector.shape[1])
    k = min(k, len(_texts))
    _, indices = index.search(np.array(vector, dtype="float32"), k)
    return [int(i) for i in indices[0] if 0 <= i < len(_texts)]


def _tokenize(text: str) -> list:
    """Whitespace-split, lowercased, with surrounding punctuation stripped
    per token (internal characters like the hyphen in "INV-88214" are kept).
    Without this, "INV-88214?" in a query and "INV-88214," in the indexed
    text would be different tokens to BM25 -- exactly the exact-match cases
    BM25 is supposed to be strong at, silently defeated by punctuation."""
    return [w.strip('.,!?;:"()[]') for w in text.lower().split()]


def _sparse_rank(query: str) -> list:
    """Return indices into _texts, best BM25 keyword match first.
    Rebuilt on demand from _texts -- cheap for the small, per-conversation
    corpora this project deals with; a larger deployment would persist the
    BM25 corpus alongside the FAISS index instead of rebuilding it."""
    if not _texts:
        return []
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(t) for t in _texts]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def retrieve_context(query: str, k: int = 3) -> list:
    """Return up to k most relevant past turns for the given query, fusing
    dense and sparse rankings with a weighted Reciprocal Rank Fusion:

        score(doc) = sum over each ranking it appears in of  W / (rank + C)

    C=60 is the constant from the original RRF paper (Cormack et al.) --
    it dampens the influence of low ranks so one ranker's #1 pick doesn't
    automatically dominate regardless of where the other ranker placed it.

    W is a per-ranker weight, NOT 1.0/1.0 by default. Plain (unweighted)
    RRF has a real failure mode confirmed empirically on this project: two
    candidates that are exact rank mirror-images of each other -- e.g. one
    ranks #1 dense/#3 sparse, the other #3 dense/#1 sparse -- produce an
    EXACT tied score under unweighted RRF, regardless of corpus size or
    pool depth (it's structural, not a tuning artifact). The tie then gets
    broken by incidental sort order rather than any real signal.
    When that happened here (two near-duplicate "ticket #XXXXX" entries,
    the correct one buried at dense-rank 2 but sitting at sparse-rank 0),
    plain RRF silently returned the wrong document. Since dense embeddings
    are a known weak point for arbitrary IDs/numbers (confirmed directly:
    see evaluation/eval_retrieval.py's exact_keyword category) while BM25
    is precisely built for exact token matches, weighting sparse slightly
    higher is a justified, evidence-based tiebreak -- not an arbitrary knob
    turned until a demo looked good.

    Returns [] if the index is empty -- callers should treat that as "no
    extra context available", not an error.
    """
    _ensure_loaded()
    if not _texts:
        return []

    pool_size = min(len(_texts), max(k * 3, k))
    dense = _dense_rank(query, pool_size)
    sparse = _sparse_rank(query)[:pool_size]

    C = 60
    DENSE_WEIGHT = 1.0
    SPARSE_WEIGHT = 1.3  # slight edge: see rationale above

    fused_scores = {}
    for rank, idx in enumerate(dense):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + DENSE_WEIGHT / (rank + C)
    for rank, idx in enumerate(sparse):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + SPARSE_WEIGHT / (rank + C)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    top_k = ranked[:k]
    return [_texts[idx] for idx, _score in top_k]


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