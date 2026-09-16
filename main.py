import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="企业财务报告问答 Agent",
	)
	parser.add_argument(
		"query",
		nargs="?",
		help="要回答的问题；不提供时进入交互模式",
	)
	parser.add_argument(
		"--vectorstore-directory",
		type=Path,
		default=PROJECT_ROOT / "data" / "vectorstore",
		help="Chroma 向量库目录",
	)
	parser.add_argument(
		"--collection-name",
		default="financial_reports",
		help="Chroma collection 名称",
	)
	parser.add_argument(
		"--embedding-model",
		default="bgem3",
		help="检索使用的 embedding 模型注册名",
	)
	parser.add_argument(
		"--model",
		default="deepseek-chat",
		help="DeepSeek 模型名称",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="LLM temperature",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=2,
		help="检索重写和答案修正的最大重试次数",
	)
	return parser


def build_graph(args: argparse.Namespace) -> Any:
	if args.max_retries < 0:
		raise ValueError("--max-retries 不能小于 0")
	if not args.vectorstore_directory.exists():
		raise FileNotFoundError(
			f"向量库目录不存在：{args.vectorstore_directory}。"
			"请先运行 scripts/ingest.py。"
		)

	from app.graph.graph import create_rag_graph
	from app.llm.model import create_deepseek_llm
	from app.retrieval.retriever import create_retriever

	llm = create_deepseek_llm(
		model=args.model,
		temperature=args.temperature,
	)
	retriever = create_retriever(
		vectorstore_directory=args.vectorstore_directory,
		collection_name=args.collection_name,
		embedding_model=args.embedding_model,
	)
	return create_rag_graph(llm, retriever, max_retries=args.max_retries)


def ask(graph: Any, text: str):
	from app.graph.state import RAGState
	from app.schemas.query import Query

	result = graph.invoke(RAGState(query=Query(text=text)))
	return result if isinstance(result, RAGState) else RAGState.model_validate(result)


def print_result(state: Any) -> None:
	print(f"\n答案：\n{state.answer or '未生成答案。'}")
	if state.context:
		print("\n参考来源：")
		seen: set[tuple[str, int]] = set()
		for document in state.context:
			key = (document.metadata.source, document.metadata.page)
			if key in seen:
				continue
			seen.add(key)
			print(
				f"- {document.metadata.source}，第 {document.metadata.page} 页"
			)
	if state.validation_result is False:
		print(f"\n提示：答案验证未通过。{state.validation_feedback}")


def interactive_loop(graph: Any) -> None:
	print("企业财务报告问答 Agent")
	print("输入 exit 或 quit 退出。")
	while True:
		try:
			text = input("\n问题> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			return
		if text.lower() in {"exit", "quit"}:
			return
		if not text:
			continue
		try:
			print_result(ask(graph, text))
		except Exception as error:
			print(f"\n处理失败：{error}", file=sys.stderr)


def main() -> int:
	args = build_parser().parse_args()
	try:
		graph = build_graph(args)
		if args.query:
			print_result(ask(graph, args.query))
		else:
			interactive_loop(graph)
		return 0
	except Exception as error:
		print(f"启动失败：{error}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
