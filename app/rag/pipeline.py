from pathlib import Path
from typing import Any, Protocol

from app.generation.context import format_context
from app.generation.prompt import build_prompt
from app.retrieval.retriever import Retriever, create_retriever
from app.schemas.query import Query


class RunnableLLM(Protocol):
	def invoke(self, input: Any) -> Any: ...


class RAGPipeline:
	def __init__(self, retriever: Retriever, llm: RunnableLLM) -> None:
		self.retriever = retriever
		self.llm = llm

	def invoke(self, query: Query | str) -> str:
		return self.answer(query)

	def run(self, query: Query | str) -> str:
		return self.answer(query)

	def answer(self, query: Query | str) -> str:
		"""Retrieve fresh documents and answer one query."""
		validated_query = query if isinstance(query, Query) else Query(text=query)
		documents = self.retriever.retrieve(validated_query)
		context = format_context(documents)
		response = self.llm.invoke(build_prompt(validated_query, context))
		return self._response_text(response)

	@staticmethod
	def _response_text(response: Any) -> str:
		content = getattr(response, "content", response)
		if isinstance(content, str):
			return content.strip()
		if isinstance(content, list):
			return "".join(
				part.get("text", "") if isinstance(part, dict) else str(part)
				for part in content
			).strip()
		return str(content).strip()


def create_rag_pipeline(
	vectorstore_directory: Path,
	collection_name: str = "financial_reports",
	embedding_model: str = "bgem3",
	llm: RunnableLLM | None = None,
	**llm_kwargs: Any,
) -> RAGPipeline:
	from app.llm.model import create_deepseek_llm

	return RAGPipeline(
		retriever=create_retriever(
			vectorstore_directory=vectorstore_directory,
			collection_name=collection_name,
			embedding_model=embedding_model,
		),
		llm=llm or create_deepseek_llm(**llm_kwargs),
	)
