from app.graph.state import RAGState
from app.schemas.rewritten_query import RewrittenQuery

from ._llm import invoke_schema


def build_query_rewrite_prompt(state: RAGState) -> str:
	if state.query is None:
		raise ValueError("RAGState.query is required before query rewrite")

	analysis = state.query_analysis.model_dump_json() if state.query_analysis else "{}"
	feedback = state.grade_feedback or "无"
	reason = "complex_query" if state.relevant_result is None else "irrelevant_documents"
	return f"""请重写下面的企业财务报告问题，并严格返回 RewrittenQuery 结构。

本次重写原因：{reason}
原始问题：{state.query.text}
Query 分析：{analysis}
上次检索相关性反馈：{feedback}

要求：将问题拆成一个或多个语义完整、可以分别进行向量检索的子问题。
每个子问题都要保留必要的指标、年份和比较关系，不要回答问题。
"""


def rewrite_query(llm, state: RAGState) -> RewrittenQuery:
	return invoke_schema(
		llm,
		RewrittenQuery,
		build_query_rewrite_prompt(state),
	)