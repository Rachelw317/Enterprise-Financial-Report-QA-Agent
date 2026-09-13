import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
	"id",
	"query",
	"retrieved_evidence",
	"generated_answer",
	"retrieval_recall",
	"answer_score",
	"groundedness_score",
	"comment",
]


def convert_json_to_csv(input_path: Path, output_path: Path) -> None:
	data = json.loads(input_path.read_text(encoding="utf-8"))
	rows = []
	for item in data:
		row = {field: item.get(field, "") for field in FIELDS}
		row["retrieved_evidence"] = json.dumps(
			row["retrieved_evidence"], ensure_ascii=False
		)
		rows.append(row)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8-sig", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=FIELDS)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert evaluation JSON results to CSV")
	parser.add_argument(
		"--input",
		type=Path,
		default=PROJECT_ROOT / "data" / "results" / "baseline_v1.json",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=PROJECT_ROOT / "data" / "results" / "baseline_v1.csv",
	)
	args = parser.parse_args()

	convert_json_to_csv(args.input, args.output)
	print(f"已转换 {args.input}，结果写入 {args.output}")


if __name__ == "__main__":
	main()