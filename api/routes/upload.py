
from pathlib import Path
from uuid import uuid4

from fastapi import status, HTTPException, UploadFile, File, APIRouter

from rag.loader import load_uploaded_file

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_SIZE = 25 #MB
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".csv",  # browsers sometimes send this for CSV
}

def validate_upload(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename",
        )

    suffix = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    # Accept either by MIME type or file extension
    if content_type in ALLOWED_TYPES:
        expected_suffix = ALLOWED_TYPES[content_type]
        if suffix not in {".pdf", ".csv"}:
            suffix = expected_suffix
        return suffix

    if suffix in {".pdf", ".csv"}:
        return suffix

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Only PDF and CSV files are allowed",
    )

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = validate_upload(file)

    # Read file once
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Optional size check: 10 MB
    max_size = MAX_UPLOAD_SIZE * 1024 * 1024
    if len(data) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Max size is 25 MB",
        )

    saved_name = f"{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    saved_path.write_bytes(data)

    # Load immediately so upload completion guarantees the file is parseable.
    documents = load_uploaded_file(saved_name)

    response = {
        "message": "Upload successful",
        "original_filename": file.filename,
        "stored_filename": saved_name,
        "content_type": file.content_type,
        "size_bytes": len(data),
        "file_type": "pdf" if suffix == ".pdf" else "csv",
        "loaded_documents": len(documents),
    }

    return response

