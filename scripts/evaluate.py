import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pipeline import create_rag_pipeline
from app.schemas.evaluation import EvaluationCase, EvaluationResult, RequiredEvidence
from app.schemas.query import Query


def load_cases(dataset_path: Path) -> list[EvaluationCase]:
	data = json.loads(dataset_path.read_text(encoding="utf-8"))
	return [EvaluationCase.model_validate(case) for case in data]


def evaluate_case(pipeline, case: EvaluationCase) -> EvaluationResult:
	query = Query(text=case.query)
	documents = pipeline.retriever.retrieve(query)
	answer = pipeline.answer_with_documents(query, documents)
	retrieved_evidence = [
		RequiredEvidence(
			source=document.metadata.source,
			page=document.metadata.page,
			data_year=document.metadata.data_year,
		)
		for document in documents
	]

	return EvaluationResult(
		id=case.id,
		query=case.query,
		retrieved_evidence=retrieved_evidence,
		generated_answer=answer,
	)


def evaluate(
	dataset_path: Path,
	output_path: Path,
	vectorstore_directory: Path,
	collection_name: str = "financial_reports",
	embedding_model: str = "bgem3",
) -> None:
	pipeline = create_rag_pipeline(
		vectorstore_directory=vectorstore_directory,
		collection_name=collection_name,
		embedding_model=embedding_model,
	)
	cases = load_cases(dataset_path)
	results = [evaluate_case(pipeline, case) for case in cases]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(
			[result.model_dump(exclude_none=True) for result in results],
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate the financial report QA pipeline")
	parser.add_argument(
		"--dataset",
		type=Path,
		default=PROJECT_ROOT / "data" / "evaluation" / "dataset.json",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=PROJECT_ROOT / "data" / "results" / "evaluation_results.json",
	)
	parser.add_argument(
		"--vectorstore-directory",
		type=Path,
		default=PROJECT_ROOT / "data" / "vectorstore",
	)
	parser.add_argument("--collection-name", default="financial_reports")
	parser.add_argument("--embedding-model", default="bgem3")
	args = parser.parse_args()

	evaluate(
		dataset_path=args.dataset,
		output_path=args.output,
		vectorstore_directory=args.vectorstore_directory,
		collection_name=args.collection_name,
		embedding_model=args.embedding_model,
	)
	print(f"已完成评测，结果写入 {args.output}")


if __name__ == "__main__":
	main()
