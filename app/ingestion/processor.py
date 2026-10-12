import json
import os
import sys
import uuid

# logfire must be configured before app module imports so spans from
# chunking/loaders/embedding are captured from the start.
import logfire
from app.config import settings

_logfire_base_url = settings.LOGFIRE_BASE_URL
if not _logfire_base_url and settings.LOGFIRE_TOKEN:
    if settings.LOGFIRE_TOKEN.startswith("pylf_v2_eu_"):
        _logfire_base_url = "https://logfire-eu.pydantic.dev"

if settings.LOGFIRE_TOKEN:
    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        service_name="enterprise-ingestion-service",
        console=logfire.ConsoleOptions(min_log_level="info"),
        advanced=logfire.AdvancedOptions(base_url=_logfire_base_url) if _logfire_base_url else None,
    )

from qdrant_client import QdrantClient
from qdrant_client.http import models

print("✅ qdrant_client imports OK")

from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text

print("✅ ingestion loader imports OK")

from app.services.retrieval.embeddings import embed_texts, get_embedding_dim

print("✅ embeddings import OK")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
print(f"🔌 Connecting to Qdrant at: {settings.QDRANT_URL}")
try:
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    print("✅ QdrantClient created OK")
except Exception as _qe:
    import traceback
    print(f"❌ QdrantClient FAILED: {_qe}")
    traceback.print_exc()
    raise


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office

                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

        except Exception as e:
            print(f"❌ Exception occurred while processing {filename}: {e}")
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        print(f"📄 Found {len(files)} files in '{dir_path}': {files}")
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            print(f"⚙️  Processing: {filename}")
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    print(f"🚀 Starting ingestion for: {base_dir}")
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        # Wipe collection if requested
        if wipe:
            print("🗑️  Wiping existing Qdrant collection...")
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    print(f"✅ Collection '{settings.QDRANT_COLLECTION}' deleted.")
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        print(f"🔍 Checking if Qdrant collection '{settings.QDRANT_COLLECTION}' exists...")
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            print("📐 Collection not found — probing embedding model for dimension...")
            dim = get_embedding_dim()
            print(f"📐 Creating Qdrant collection with dim={dim}...")
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim, Cosine).")
            logfire.info(f"Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim, Cosine).")
        else:
            print(f"✅ Collection '{settings.QDRANT_COLLECTION}' already exists.")

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"📂 Sub-directories found: {subdirs if subdirs else 'None — treating as flat directory'}")

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"
            print(f"📁 Processing '{base_dir}' as source_type='{source_type}'")
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
                print(f"📁 Processing sub-dir '{subdir}' as source_type='{source_type}'")
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    print("🟢 __main__ reached")
    print(f"   QDRANT_URL: {settings.QDRANT_URL}")
    print(f"   QDRANT_COLLECTION: {settings.QDRANT_COLLECTION}")
    print(f"   GEMINI_API_KEY set: {bool(settings.GEMINI_API_KEY)}")
    print(f"   LOGFIRE_TOKEN set: {bool(settings.LOGFIRE_TOKEN)}")

    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    try:
        run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
        logfire.info("Ingestion job completed.")
        logfire.force_flush()
        print("✅ Ingestion complete!")
    except Exception as e:
        import traceback
        print(f"\n❌ Ingestion CRASHED: {e}")
        traceback.print_exc()

