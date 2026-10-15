import logfire
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.embeddings.providers import register_embedding_provider

from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
from app.guardrails.nemo_embeddings import ThrottledGeminiEmbeddingModel, _colang_hash
from app.gateway.client import get_langchain_llm


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.

    LLM:        ChatGroq llama-3.1-8b-instant  (fast intent classification)
    Embeddings: ThrottledGeminiEmbeddingModel   (engine="throttled_gemini")
                - Reuses app Gemini embeddings (no new downloads)
                - Throttled to 0.7s/sentence to avoid 429 rate limits
                - Results cached to disk; only re-embeds when colang_rules.py
                  changes (detected via MD5 hash)

    The heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    # --- 1. Register custom embeddings BEFORE building config ---------------
    #   This adds "throttled_gemini" to NeMo's EmbeddingProviderRegistry so
    #   BasicEmbeddingsIndex can find it when the config specifies
    #   engine="throttled_gemini".
    register_embedding_provider(ThrottledGeminiEmbeddingModel)

    # --- 2. Guard LLM (via Portkey gateway — same path that works for RAG) ---
    #   Direct ChatGroq calls fail with 404 (GROQ_API_KEY is routed through
    #   Portkey, not used for direct Groq API access). get_langchain_llm()
    #   returns a ChatOpenAI pointed at PORTKEY_GATEWAY_URL which is confirmed
    #   working for the main RAG pipeline.
    guard_llm = get_langchain_llm(feature="guardrails")

    # --- 3. Build NeMo config from Colang + YAML strings --------------------
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    # --- 4. Inject embeddings model into config.models ----------------------
    #   LLMRails.__init__ scans config.models for type="embeddings" and sets
    #   _default_embedding_engine = model.engine and
    #   _default_embedding_params = model.parameters.
    #   BasicEmbeddingsIndex then calls init_embedding_model("throttled_gemini")
    #   which looks up ThrottledGeminiEmbeddingModel from the registry and
    #   instantiates it, passing colang_hash as a keyword arg so the disk
    #   cache can detect when colang_rules.py has changed.
    ch = _colang_hash(COLANG_CONTENT)
    config.models.append(
        Model(
            type="embeddings",
            engine="throttled_gemini",
            model="gemini-embedding-2-preview",
            parameters={"colang_hash": ch},
        )
    )

    # --- 5. Build rails -----------------------------------------------------
    _rails = LLMRails(config, llm=guard_llm)

    logfire.info("NeMo Guardrails initialised (Portkey/llama-3.3-70b-versatile + throttled Gemini embeddings)")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) -- a rail fired; return this response immediately,
                                  skip the RAG pipeline entirely.
        (False, None)          -- message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("Guardrails not initialised -- skipping gate.")
        return False, None

    with logfire.span("Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} -- extract text.
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info("Guardrails fired", query=message[:80])
            return True, content

        logfire.info("Guardrails passed")
        return False, None