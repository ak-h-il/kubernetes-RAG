import logfire
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.embeddings.providers import register_embedding_provider

from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
from app.guardrails.nemo_embeddings import ThrottledGeminiEmbeddingModel, _colang_hash
from app.guardrails.embedding_guard import build_guard_index, guard_by_embedding
from app.gateway.client import get_langchain_llm


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the guardrail system at app startup.

    PRIMARY gate:  embedding_guard.py
        Pure cosine-similarity check against embedded user-message examples.
        No LLM call. Deterministic. Fast (~200 ms — one embed_query call).
        Vectors cached to disk; only re-embeds when colang_rules.py changes.

    FALLBACK gate: NeMo LLMRails (kept for future extensibility)
        Only reached if the embedding gate passes. NeMo is NOT used for
        intent classification (its single-call LLM path ignores colang rules
        in practice). It is initialised here but guard() calls the embedding
        gate first and short-circuits before _rails.generate() is ever called
        for blocked intents.
    """
    global _rails

    # --- 1. Build pure-embedding guard index --------------------------------
    #   Embeds all USER_INTENTS examples once, caches to guard_index.pkl.
    #   Subsequent startups load from cache in ~0.05 s.
    build_guard_index()

    # --- 2. Register custom embeddings for NeMo (kept for fallback) ---------
    register_embedding_provider(ThrottledGeminiEmbeddingModel)

    # --- 3. Build NeMo config -----------------------------------------------
    guard_llm = get_langchain_llm(feature="guardrails")
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    ch = _colang_hash(COLANG_CONTENT)
    config.models.append(
        Model(
            type="embeddings",
            engine="throttled_gemini",
            model="gemini-embedding-2-preview",
            parameters={"colang_hash": ch},
        )
    )

    _rails = LLMRails(config, llm=guard_llm)

    logfire.info("Guardrails initialised (embedding gate + NeMo fallback)")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the guardrail gate.

    Step 1 — Pure embedding similarity (no LLM):
        Embeds the message, compares against all USER_INTENTS examples.
        If cosine similarity >= SIMILARITY_THRESHOLD, returns the predefined
        BOT_RESPONSES string immediately.

    Step 2 — NeMo LLMRails (fallback, currently not reached for blocked intents):
        Only runs if the embedding gate passes. Kept for extensibility.

    Returns:
        (True,  response) -- intent matched; return response, skip RAG.
        (False, None)     -- message is clean; proceed to LangGraph.
    """
    # --- Step 1: fast embedding gate ----------------------------------------
    fired, response = guard_by_embedding(message)
    if fired:
        logfire.info("Guardrails fired (embedding gate)", query=message[:80])
        return True, response

    logfire.info("Guardrails passed")
    return False, None