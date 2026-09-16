from app.schemas.query import Query
from app.schemas.query_analyze import QueryAnalysis

from ._llm import invoke_schema


def build_query_analysis_prompt(query: Query) -> str:
	return f"""请分析下面的企业财务报告问题，并严格返回 QueryAnalysis 结构。

判断问题类型、涉及的指标和年份、需要执行的计算操作。
如果问题包含多个相互独立的指标、年份或计算步骤，请将 is_complex 设为 true，
并在 retrieval_queries 中给出可独立检索的子问题；否则给出一个检索问题。

用户问题：
{query.text}
"""


def analyze_query(llm, query: Query) -> QueryAnalysis:
	return invoke_schema(
		llm,
		QueryAnalysis,
		build_query_analysis_prompt(query),
	)