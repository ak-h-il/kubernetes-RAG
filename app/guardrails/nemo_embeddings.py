"""
Custom NeMo Guardrails embedding provider — registered as engine "throttled_gemini".

HOW IT PLUGS IN (NeMo 0.24.1):
  1. rails.py calls register_embedding_provider(ThrottledGeminiEmbeddingModel)
     before building RailsConfig.
  2. A Model(type="embeddings", engine="throttled_gemini", ...) entry is
     appended to config.models so LLMRails picks it up.
  3. LLMRails.__init__ sees type="embeddings", sets _default_embedding_engine
     = "throttled_gemini".
  4. BasicEmbeddingsIndex._init_model() calls init_embedding_model() which
     looks up "throttled_gemini" in EmbeddingProviderRegistry and instantiates
     this class. NeMo never touches FastEmbed again.

FIX SUMMARY:
  Problem 1 — FastEmbed downloads all-MiniLM-L6-v2-onnx from HuggingFace
               -> Windows firewall drops connection -> 66 s retry loop.
  Problem 2 — Switching to Gemini: NeMo burst-embeds all 40+ colang sentences
               at once -> 429 RESOURCE_EXHAUSTED on the free tier.

  Solution  — Subclass EmbeddingModel, reuse app's embed_query() (which has
               retry logic), throttle to 1 call per _DELAY_SEC, cache results
               to disk keyed by MD5 hash of COLANG_CONTENT so the 40-sentence
               index is only built ONCE (first startup). All later restarts
               load from disk in ~0.05 s.
"""

import asyncio
import hashlib
import pickle
import time
import logfire
from pathlib import Path
from typing import List

from nemoguardrails.embeddings.providers.base import EmbeddingModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 0.7 s/call = ~85 req/min, safely under Gemini free-tier cap (100 req/min).
_DELAY_SEC = 0.7

# Cache file lives next to this module.
_CACHE_PATH = Path(__file__).parent / "guardrails_index.pkl"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _colang_hash(colang_content: str) -> str:
    """MD5 of COLANG_CONTENT — detects edits to colang_rules.py."""
    return hashlib.md5(colang_content.encode()).hexdigest()


def _load_cache(expected_hash: str) -> List[List[float]] | None:
    """Return cached vectors if hash matches, else None."""
    if not _CACHE_PATH.exists():
        return None
    try:
        with _CACHE_PATH.open("rb") as f:
            data = pickle.load(f)
        if data.get("hash") == expected_hash:
            logfire.info(
                "NeMo colang index loaded from disk cache",
                path=str(_CACHE_PATH),
                sentences=len(data["vectors"]),
            )
            return data["vectors"]
        logfire.info("NeMo colang cache stale (colang_rules.py changed) — re-embedding.")
    except Exception as e:
        logfire.warning("NeMo colang cache corrupt — re-embedding.", error=str(e))
    return None


def _save_cache(colang_hash: str, vectors: List[List[float]]) -> None:
    """Persist vectors + hash to disk for future startups."""
    try:
        with _CACHE_PATH.open("wb") as f:
            pickle.dump({"hash": colang_hash, "vectors": vectors}, f)
        logfire.info(
            "NeMo colang index saved to disk cache",
            path=str(_CACHE_PATH),
            sentences=len(vectors),
        )
    except Exception as e:
        logfire.warning("NeMo colang cache could not be saved.", error=str(e))


# ---------------------------------------------------------------------------
# Custom EmbeddingModel registered with NeMo's provider registry
# ---------------------------------------------------------------------------

class ThrottledGeminiEmbeddingModel(EmbeddingModel):
    """
    NeMo-registered embedding provider (engine="throttled_gemini").

    NeMo instantiates this via:
        ThrottledGeminiEmbeddingModel(
            embedding_model="gemini-embedding-2-preview",
            colang_hash="<md5>",   # passed through model.parameters
        )

    encode() is called by BasicEmbeddingsIndex for both:
      - Index building at startup: many documents, throttled + cached to disk.
      - Query matching at inference: single document, no throttle needed.
    """

    engine_name = "throttled_gemini"

    def __init__(self, embedding_model: str, colang_hash: str = "", **kwargs) -> None:
        self._model_name = embedding_model
        self._colang_hash = colang_hash

        # Lazy import: embeddings.py must initialise AFTER logfire is ready.
        from app.services.retrieval.embeddings import embed_query, get_embedding_dim

        self._embed_fn = embed_query

        try:
            self._embedding_size = get_embedding_dim()
        except Exception:
            self._embedding_size = 3072  # Gemini default; corrected on first call.

        logfire.info(
            "NeMo ThrottledGeminiEmbeddingModel ready",
            engine=self.engine_name,
            model=self._model_name,
            dim=self._embedding_size,
            throttle_sec=_DELAY_SEC,
        )

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    # ------------------------------------------------------------------
    # encode() — sync, called by BasicEmbeddingsIndex
    # ------------------------------------------------------------------
    def encode(self, documents: List[str]) -> List[List[float]]:
        """
        Encode a batch of documents.

        Single document  → query at inference time (no throttle, no cache).
        Multiple documents → index build at startup (throttled + disk cache).
        """
        if len(documents) == 1:
            # Fast path: per-request query embed.
            with logfire.span("NeMo embed_query"):
                vec = self._embed_fn(documents[0])
            self._embedding_size = len(vec)
            return [vec]

        # Index-build path: check disk cache first.
        cached = _load_cache(self._colang_hash)
        if cached is not None:
            if cached:
                self._embedding_size = len(cached[0])
            return cached

        # Cache miss: throttled embed of all colang sentences.
        total = len(documents)
        logfire.info(
            "NeMo building colang intent index (cache miss)",
            sentences=total,
            estimated_sec=round(total * _DELAY_SEC, 1),
        )

        results: List[List[float]] = []
        for i, sentence in enumerate(documents):
            with logfire.span("NeMo index sentence", index=i + 1, total=total):
                vec = self._embed_fn(sentence)
            results.append(vec)
            if i < total - 1:
                time.sleep(_DELAY_SEC)

        if results:
            self._embedding_size = len(results[0])

        _save_cache(self._colang_hash, results)
        logfire.info("NeMo colang intent index ready", sentences=total)
        return results

    # ------------------------------------------------------------------
    # encode_async() — async wrapper required by EmbeddingModel ABC
    # ------------------------------------------------------------------
    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, documents)
