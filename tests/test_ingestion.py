import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents


def main() -> None:
	documents = load_documents()
	chunks = split_documents(documents)

	print(f"Loaded documents: {len(documents)}")
	print("\n========== First 10 Documents ==========")
	for index, document in enumerate(documents[:10], start=1):
		print(f"\n--- Document {index} ---")
		print(f"Metadata: {document.metadata.model_dump()}")
		print(f"Content:\n{document.content}")

	print(f"\nGenerated chunks: {len(chunks)}")
	print("\n========== First 50 Chunks ==========")
	for index, chunk in enumerate(chunks[:50], start=1):
		print(f"\n--- Chunk {index} ---")
		print(f"Metadata: {chunk.metadata.model_dump()}")
		print(f"Content:\n{chunk.content}")


if __name__ == "__main__":
	main()
