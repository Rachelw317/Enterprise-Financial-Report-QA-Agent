import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pipeline import build_chunks
from app.retrieval.embedding import create_embeddings
from app.retrieval.vectorstore import VectorStore


def _chunk_id(source: str, page: int, content: str) -> str:
	value = f"{source}:{page}:{content}".encode("utf-8")
	return hashlib.sha256(value).hexdigest()


def ingest(
	pdf_directory: Path,
	vectorstore_directory: Path,
	collection_name: str = "financial_reports",
	embedding_model: str = "bgem3",
	chunk_size: int = 1000,
	chunk_overlap: int = 100,
) -> int:
	chunks = build_chunks(
		pdf_directory=pdf_directory,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	if not chunks:
		print(f"No PDF documents found in {pdf_directory}")
		return 0

	embeddings = create_embeddings(embedding_model)
	store = VectorStore(
		persist_directory=vectorstore_directory,
		collection_name=collection_name,
		embeddings=embeddings,
	)
	ids = [
		_chunk_id(chunk.metadata.source, chunk.metadata.page, chunk.content)
		for chunk in chunks
	]
	store.add_documents(chunks, ids=ids)

	print(f"Loaded chunks: {len(chunks)}")
	print(f"Vector store: {vectorstore_directory}")
	print(f"Collection count: {store.count()}")
	return len(chunks)


def main() -> None:
	parser = argparse.ArgumentParser(description="Ingest PDF reports into the vector store")
	parser.add_argument(
		"--pdf-directory",
		type=Path,
		default=PROJECT_ROOT / "data" / "raw",
	)
	parser.add_argument(
		"--vectorstore-directory",
		type=Path,
		default=PROJECT_ROOT / "data" / "vectorstore",
	)
	parser.add_argument("--collection-name", default="financial_reports")
	parser.add_argument("--embedding-model", default="bgem3")
	parser.add_argument("--chunk-size", type=int, default=1000)
	parser.add_argument("--chunk-overlap", type=int, default=100)
	args = parser.parse_args()

	ingest(
		pdf_directory=args.pdf_directory,
		vectorstore_directory=args.vectorstore_directory,
		collection_name=args.collection_name,
		embedding_model=args.embedding_model,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)


if __name__ == "__main__":
	main()
