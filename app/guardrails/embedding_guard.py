"""
Pure-embedding guardrail gate — replaces NeMo's broken single-call LLM path.

HOW IT WORKS:
  At startup, build_guard_index() embeds every user-message example from
  colang_rules.py and stores them as (intent, vector) pairs.

  At request time, guard_by_embedding() embeds the user message once and
  computes cosine similarity against every stored vector.  If the best match
  exceeds SIMILARITY_THRESHOLD the corresponding bot response is returned
  immediately — no LLM call, no NeMo index bugs.

WHY THIS INSTEAD OF NEMO:
  generate_intent_steps_message() always calls the LLM regardless of the
  embeddings_only flag — it only uses embeddings to build few-shot examples
  for the LLM prompt.  The LLM then ignores those examples and answers the
  question anyway (e.g. tells a joke).  This implementation skips the LLM
  entirely for intent classification.
"""

import hashlib
import pickle
import time
import numpy as np
import logfire
from pathlib import Path
from typing import Optional

from app.guardrails.colang_rules import (
    USER_INTENTS,
    BOT_RESPONSES,
    BLOCKING_INTENTS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Cosine similarity threshold.  1.0 = exact match, 0.0 = unrelated.
# 0.75 is tight enough to avoid false positives but catches paraphrases well.
SIMILARITY_THRESHOLD = 0.75

_DELAY_SEC = 0.7          # throttle between Gemini embedding calls
_CACHE_PATH = Path(__file__).parent / "guard_index.pkl"

# ---------------------------------------------------------------------------
# In-memory index: list of (intent_name, unit_vector)
# ---------------------------------------------------------------------------
_index: list[tuple[str, np.ndarray]] = []
_index_ready = False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _content_hash() -> str:
    """MD5 of the intent/response tables — detects colang_rules.py edits."""
    raw = str(USER_INTENTS) + str(BOT_RESPONSES)
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(expected_hash: str, expected_count: int) -> list[tuple[str, list[float]]] | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        with _CACHE_PATH.open("rb") as f:
            data = pickle.load(f)
        if data.get("hash") == expected_hash and len(data.get("items", [])) == expected_count:
            logfire.info("Guard index loaded from cache", sentences=expected_count)
            return data["items"]
        logfire.info("Guard cache miss (stale or size mismatch) — re-embedding.")
    except Exception as e:
        logfire.warning("Guard cache corrupt — re-embedding.", error=str(e))
    return None


def _save_cache(h: str, items: list[tuple[str, list[float]]]) -> None:
    try:
        with _CACHE_PATH.open("wb") as f:
            pickle.dump({"hash": h, "items": items}, f)
        logfire.info("Guard index saved to cache", sentences=len(items))
    except Exception as e:
        logfire.warning("Guard cache could not be saved.", error=str(e))


# ---------------------------------------------------------------------------
# Build index at startup
# ---------------------------------------------------------------------------

def build_guard_index() -> None:
    """
    Embed every user-message example and store as unit vectors.
    Called once from initialize_rails() at app startup.
    """
    global _index, _index_ready

    from app.services.retrieval.embeddings import embed_query

    # Flatten USER_INTENTS into (intent, sentence) pairs
    pairs: list[tuple[str, str]] = []
    for intent, sentences in USER_INTENTS.items():
        for s in sentences:
            pairs.append((intent, s))

    h = _content_hash()
    cached = _load_cache(h, len(pairs))

    if cached is not None:
        _index = [(intent, np.array(vec, dtype=np.float32)) for intent, vec in cached]
        # Normalise (cache stores raw vectors)
        _index = [(intent, v / (np.linalg.norm(v) or 1.0)) for intent, v in _index]
        _index_ready = True
        return

    logfire.info(
        "Building guard embedding index (cache miss)",
        sentences=len(pairs),
        estimated_sec=round(len(pairs) * _DELAY_SEC, 1),
    )

    raw: list[tuple[str, list[float]]] = []
    for i, (intent, sentence) in enumerate(pairs):
        with logfire.span("Guard index sentence", index=i + 1, total=len(pairs)):
            vec = embed_query(sentence)
        raw.append((intent, vec))
        if i < len(pairs) - 1:
            time.sleep(_DELAY_SEC)

    _save_cache(h, raw)

    _index = []
    for intent, vec in raw:
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        _index.append((intent, v / (norm or 1.0)))

    _index_ready = True
    logfire.info("Guard embedding index ready", sentences=len(_index))


# ---------------------------------------------------------------------------
# Per-request classification
# ---------------------------------------------------------------------------

def guard_by_embedding(message: str) -> tuple[bool, str | None]:
    """
    Returns (fired, response_text) — purely from cosine similarity.
    No LLM call.  Average latency: one embed_query() call (~200 ms).
    """
    if not _index_ready or not _index:
        logfire.warning("Guard index not ready — skipping embedding gate.")
        return False, None

    from app.services.retrieval.embeddings import embed_query

    with logfire.span("Guard embed_query"):
        q = np.array(embed_query(message), dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # Cosine similarity against every stored unit vector
    best_score = -1.0
    best_intent = None
    for intent, v in _index:
        score = float(np.dot(q, v))
        if score > best_score:
            best_score = score
            best_intent = intent

    logfire.info(
        "Guard similarity result",
        best_intent=best_intent,
        best_score=round(best_score, 4),
        threshold=SIMILARITY_THRESHOLD,
    )

    if best_score >= SIMILARITY_THRESHOLD and best_intent in BLOCKING_INTENTS:
        response = BOT_RESPONSES.get(best_intent, "I can't help with that.")
        logfire.info("Guard fired", intent=best_intent, score=round(best_score, 4))
        return True, response

    return False, None
