import getpass
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec, Pinecone

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = Path("uploads")


class UnsupportedFileTypeError(ValueError):
    """Raised when trying to load an unsupported uploaded file type."""


def _resolve_uploaded_file(stored_filename: str) -> Path:
    """Resolve and validate a filename returned by the upload route."""
    file_path = UPLOAD_DIR / stored_filename

    # Block path traversal and enforce that file exists in upload directory.
    if file_path.parent != UPLOAD_DIR:
        raise ValueError("Invalid uploaded filename")

    if not file_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {stored_filename}")

    return file_path


def load_uploaded_file(stored_filename: str) -> list[Document]:
    """
    Load a file saved by the /upload route.

    Args:
        stored_filename: The `stored_filename` value returned by the upload API.

    Returns:
        A list of LangChain Document objects.
    """
    file_path = _resolve_uploaded_file(stored_filename)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyMuPDFLoader(str(file_path)).load()

    if suffix == ".csv":
        return CSVLoader(file_path=str(file_path)).load()

    raise UnsupportedFileTypeError(
        f"Unsupported file extension '{suffix}'. Only .pdf and .csv are supported."
    )

def data_extraction(stored_filename: str):
    try:
        data = load_uploaded_file(stored_filename)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(data)
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "developer-quickstart-py"
        print(os.environ["PINECONE_API_KEY"])
        # Compute one embedding to get the true dimension
        test_vector = embeddings.embed_query("dimension check")
        dimension = len(test_vector)
        print(pc.list_indexes())
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={
                    "environment": "development"
                }
            )

        index = pc.Index(index_name)

        vectors = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            vector = embeddings.embed_query(text)
            vectors.append(
                {
                    "id": f"{stored_filename}-{i}",
                    "values": vector,
                    "metadata": {
                        "text": text,
                        **chunk.metadata,
                    },
                }
            )

        # Upsert in batches
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[start:start + batch_size])

        return "Success: Embedded and stored in Pinecone."
    except UnsupportedFileTypeError as e:
        return e


