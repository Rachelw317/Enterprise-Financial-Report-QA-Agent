from pathlib import Path

from app.ingestion.loader import load_documents, raw_pdf_file_path
from app.ingestion.splitter import split_documents
from app.schemas.document import Document


def build_chunks(
	pdf_directory: Path = raw_pdf_file_path,
	chunk_size: int = 1000,
	chunk_overlap: int = 100,
) -> list[Document]:
	documents = load_documents(pdf_directory)
	return split_documents(
		documents,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
