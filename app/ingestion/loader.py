import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from app.schemas.document import Document, DocumentMetadata

raw_pdf_file_path = Path(__file__).parent.parent.parent / "data" / "raw"


def _extract_year_from_filename(file_path: Path) -> int:
	match = re.search(r"(?<!\d)(20\d{2})(?!\d)", file_path.stem)
	if match is None:
		raise ValueError(f"Cannot determine document year from filename: {file_path.name}")
	return int(match.group(1))


def _extract_data_years(content: str) -> list[int]:
	years = re.findall(r"(?<!\d)(20\d{2})(?!\d)\s*年", content)
	return sorted({int(year) for year in years}, reverse=True)


def load_documents(pdf_directory: Path = raw_pdf_file_path) -> list[Document]:
	documents: list[Document] = []

	for pdf_path in sorted(pdf_directory.glob("*.pdf")):
		document_year = _extract_year_from_filename(pdf_path)

		for page_document in PyMuPDFLoader(str(pdf_path)).load():
			content = page_document.page_content
			metadata = DocumentMetadata(
				source=pdf_path.name,
				page=page_document.metadata["page"] + 1,
				document_year=document_year,
				data_year=_extract_data_years(content),
			)
			documents.append(Document(content=content, metadata=metadata))

	return documents
