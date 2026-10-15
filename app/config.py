import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_BASE_URL = os.getenv("LOGFIRE_BASE_URL")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "first-key"    # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "second-key"  # fallback: @brag/llama-3.1-8b-instant




settings = Settings()