import csv
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

UPLOAD_DIR = Path("uploads").resolve()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "developer-quickstart-py")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

# Keep this conservative to reduce timeout / overload risk.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))

# Retry config
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))
EMBED_BASE_DELAY = float(os.getenv("EMBED_BASE_DELAY", "2.0"))
EMBED_MAX_DELAY = float(os.getenv("EMBED_MAX_DELAY", "30.0"))

# Optional per-upload throttling between embedding batch calls (seconds).
EMBED_MIN_INTERVAL_SECONDS = float(os.getenv("EMBED_MIN_INTERVAL_SECONDS", "0.5"))


class UnsupportedFileTypeError(ValueError):
    pass


class MissingEnvironmentVariableError(EnvironmentError):
    pass


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise MissingEnvironmentVariableError(
            f"Missing required environment variable: {var_name}"
        )
    return value


def _resolve_uploaded_file(stored_filename: str) -> Path:
    file_path = (UPLOAD_DIR / stored_filename).resolve()

    try:
        file_path.relative_to(UPLOAD_DIR)
    except ValueError as exc:
        raise ValueError("Invalid uploaded filename") from exc

    if not file_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {stored_filename}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Uploaded path is not a file: {stored_filename}")

    return file_path

def fast_csv_loader(file_path: str):
    docs = []
    with open(file_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = " | ".join(f"{k}: {v}" for k, v in row.items())
            docs.append(Document(page_content=text, metadata={"row": i}))
    return docs

def load_uploaded_file(stored_filename: str) -> list[Document]:
    file_path = _resolve_uploaded_file(stored_filename)
    suffix = file_path.suffix.lower()

    logger.info("Loading uploaded file: %s", file_path.name)

    if suffix == ".pdf":
        docs = PyMuPDFLoader(str(file_path)).load()
    elif suffix == ".csv":
        docs = fast_csv_loader(str(file_path))
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{suffix}'. Only .pdf and .csv are supported."
        )

    logger.info("Loaded %d document(s) from %s", len(docs), file_path.name)
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


def get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    _require_env("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_pinecone_client() -> Pinecone:
    _require_env("PINECONE_API_KEY")
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index(pc: Pinecone, embeddings: GoogleGenerativeAIEmbeddings) -> None:
    if pc.has_index(INDEX_NAME):
        logger.info("Pinecone index already exists: %s", INDEX_NAME)
        return

    logger.info("Creating Pinecone index: %s", INDEX_NAME)
    test_vector = embeddings.embed_query("dimension check")
    dimension = len(test_vector)

    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
        deletion_protection="disabled",
        tags={"environment": "development"},
    )

    logger.info("Created Pinecone index '%s' with dimension=%d", INDEX_NAME, dimension)


def _is_retryable_embedding_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "429",
        "resource exhausted",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "503",
        "unavailable",
        "service is currently unavailable",
        "deadline exceeded",
        "timeout",
        "temporarily unavailable",
        "internal",
    ]
    return any(marker in message for marker in retry_markers)


def embed_documents_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
) -> list[list[float]]:
    last_exc: Exception | None = None

    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            logger.info(
                "Embedding batch of %d chunk(s), attempt %d/%d",
                len(texts),
                attempt,
                EMBED_MAX_RETRIES,
            )
            return embeddings.embed_documents(texts)

        except GoogleGenerativeAIError as exc:
            last_exc = exc
            if not _is_retryable_embedding_error(exc) or attempt == EMBED_MAX_RETRIES:
                logger.exception("Embedding failed permanently")
                raise

            delay = min(EMBED_BASE_DELAY * (2 ** (attempt - 1)), EMBED_MAX_DELAY)
            delay += random.uniform(0, 1.0)

            logger.warning(
                "Transient embedding failure on attempt %d/%d: %s. Retrying in %.2fs",
                attempt,
                EMBED_MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

        except Exception as exc:
            last_exc = exc
            logger.exception("Unexpected embedding failure")
            raise

    if last_exc:
        raise last_exc

    raise RuntimeError("Embedding failed unexpectedly without an exception.")


def build_vectors(
    chunks: list[Document],
    stored_filename: str,
    embeddings: GoogleGenerativeAIEmbeddings,
) -> list[dict[str, Any]]:
    texts = [chunk.page_content for chunk in chunks]
    vectors: list[dict[str, Any]] = []
    last_embed_request_at = 0.0

    logger.info("Generating embeddings for %d chunk(s)", len(texts))

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_chunks = chunks[start : start + EMBED_BATCH_SIZE]
        batch_texts = texts[start : start + EMBED_BATCH_SIZE]

        if EMBED_MIN_INTERVAL_SECONDS > 0:
            elapsed = time.monotonic() - last_embed_request_at
            if elapsed < EMBED_MIN_INTERVAL_SECONDS:
                sleep_for = EMBED_MIN_INTERVAL_SECONDS - elapsed
                logger.info(
                    "Throttling embedding requests for %.2fs to reduce rate-limit risk",
                    sleep_for,
                )
                time.sleep(sleep_for)

        batch_vectors = embed_documents_with_retry(embeddings, batch_texts)
        last_embed_request_at = time.monotonic()

        for offset, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
            i = start + offset
            vectors.append(
                {
                    "id": f"{stored_filename}-{i}",
                    "values": vector,
                    "metadata": {
                        "source_file": stored_filename,
                        "chunk_index": i,
                        "text": chunk.page_content,
                        **chunk.metadata,
                    },
                }
            )

    logger.info("Built %d vector(s)", len(vectors))
    return vectors


def upsert_vectors(
    pc: Pinecone,
    vectors: list[dict[str, Any]],
    namespace: str = PINECONE_NAMESPACE,
) -> int:
    index = pc.Index(INDEX_NAME)
    total_upserted = 0

    logger.info(
        "Upserting %d vector(s) into index='%s', namespace='%s'",
        len(vectors),
        INDEX_NAME,
        namespace,
    )

    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    logger.info("Finished upserting %d vector(s)", total_upserted)
    return total_upserted


def data_extraction(stored_filename: str) -> dict[str, Any]:
    try:
        logger.info("Starting data extraction for file: %s", stored_filename)

        documents = load_uploaded_file(stored_filename)
        chunks = split_documents(documents)

        if not chunks:
            logger.warning("No chunks were produced for file: %s", stored_filename)
            return {
                "status": "warning",
                "message": "No content found to embed.",
                "file": stored_filename,
                "documents": len(documents),
                "chunks": 0,
                "upserted": 0,
            }

        embeddings = get_embeddings_client()
        pc = get_pinecone_client()

        ensure_index(pc, embeddings)
        vectors = build_vectors(chunks, stored_filename, embeddings)
        upserted_count = upsert_vectors(pc, vectors)

        logger.info("Completed data extraction for file: %s", stored_filename)

        return {
            "status": "success",
            "message": "Embedded and stored in Pinecone.",
            "file": stored_filename,
            "documents": len(documents),
            "chunks": len(chunks),
            "upserted": upserted_count,
            "index_name": INDEX_NAME,
            "namespace": PINECONE_NAMESPACE,
        }

    except UnsupportedFileTypeError as exc:
        logger.warning("Unsupported file type for '%s': %s", stored_filename, exc)
        return {"status": "error", "message": str(exc), "file": stored_filename}

    except FileNotFoundError as exc:
        logger.warning("File not found: %s", exc)
        return {"status": "error", "message": str(exc), "file": stored_filename}

    except MissingEnvironmentVariableError as exc:
        logger.error("Configuration error: %s", exc)
        return {"status": "error", "message": str(exc), "file": stored_filename}

    except Exception:
        logger.exception("Unexpected failure while processing file: %s", stored_filename)
        return {
            "status": "error",
            "message": "Unexpected error while embedding and storing data.",
            "file": stored_filename,
        }
