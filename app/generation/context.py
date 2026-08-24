from collections.abc import Sequence

from app.schemas.context import Context
from app.schemas.document import Document


def format_context(documents: Sequence[Document]) -> str:
	"""Render retrieved documents as a source-aware context for the LLM."""
	validated_context = Context(documents=list(documents))
	if not validated_context.documents:
		return "未检索到相关财务报告内容。"

	sections: list[str] = []
	for index, document in enumerate(validated_context.documents, start=1):
		metadata = document.metadata
		data_years = ", ".join(str(year) for year in metadata.data_year) or "未知"
		sections.append(
			f"[{index}] 来源: {metadata.source}; 页码: {metadata.page}; "
			f"报告年份: {metadata.document_year}; 数据年份: {data_years}\n"
			f"{document.content.strip()}"
		)

	return "\n\n".join(sections)


build_context = format_context
