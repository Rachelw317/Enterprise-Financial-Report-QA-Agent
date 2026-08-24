import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pipeline import create_rag_pipeline
from app.schemas.query import Query


def chat(
	vectorstore_directory: Path,
	collection_name: str = "financial_reports",
	embedding_model: str = "bgem3",
) -> None:
	pipeline = create_rag_pipeline(
		vectorstore_directory=vectorstore_directory,
		collection_name=collection_name,
		embedding_model=embedding_model,
	)

	print("Financial report QA")
	print("请输入问题，输入 exit 或 quit 退出。")
	while True:
		try:
			text = input("\nQuery> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break

		if text.lower() in {"exit", "quit"}:
			break
		if not text:
			continue

		try:
			answer = pipeline.answer(Query(text=text))
		except Exception as error:
			print(f"\n回答失败: {error}")
			continue

		print(f"\nAnswer> {answer}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Query financial report chunks from the vector store"
	)
	parser.add_argument(
		"--vectorstore-directory",
		type=Path,
		default=PROJECT_ROOT / "data" / "vectorstore",
	)
	parser.add_argument("--collection-name", default="financial_reports")
	parser.add_argument("--embedding-model", default="bgem3")
	args = parser.parse_args()

	chat(
		vectorstore_directory=args.vectorstore_directory,
		collection_name=args.collection_name,
		embedding_model=args.embedding_model,
	)


if __name__ == "__main__":
	main()
