from app.graph.state import RAGState
from app.schemas.grade import GradeResult

from ._llm import invoke_schema


def build_grade_prompt(state: RAGState) -> str:
	query = state.query.text if state.query else ""
	documents = "\n\n".join(document.content for document in state.context)
	return f"""请判断检索结果是否足以支持回答用户问题，并严格返回 GradeResult 结构。

用户问题：{query}

检索结果：
{documents or '未检索到文档'}

如果结果缺少关键指标、年份或比较所需证据，relevant 设为 false，并给出具体改写建议。
"""


def grade_documents(llm, state: RAGState) -> GradeResult:
	return invoke_schema(llm, GradeResult, build_grade_prompt(state))