
from pathlib import Path
from uuid import uuid4

from fastapi import status, HTTPException, UploadFile, File, APIRouter


router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_SIZE = 35 #MB
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/vnd.ms-excel": ".csv",  # browsers sometimes send this for CSV
}
def get_docling_converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True
    pdf_options.do_table_structure = True
    pdf_options.table_structure_options = TableStructureOptions(do_cell_matching=True)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.CSV],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )
def extract_with_docling(file_path: Path, suffix: str) -> dict:
    try:
        doc_converter = get_docling_converter()
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document

        extracted = {"markdown": doc.export_to_markdown(), "text": doc.export_to_text(),
                     "structured": doc.export_to_dict(), "num_pages": getattr(doc, "num_pages", lambda: None)(),
                     "file_type": "pdf" if suffix == ".pdf" else "csv"}

        # Optional metadata

        return extracted

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract document data: {str(e)}",
        ) from e

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
            detail="File too large. Max size is 10 MB",
        )

    saved_name = f"{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    saved_path.write_bytes(data)

    extracted = extract_with_docling(saved_path, suffix)

    response = {
        "message": "Upload successful",
        "original_filename": file.filename,
        "stored_filename": saved_name,
        "content_type": file.content_type,
        "size_bytes": len(data),
        "file_type": "pdf" if suffix == ".pdf" else "csv",
        "extracted": extracted,
    }

    return response