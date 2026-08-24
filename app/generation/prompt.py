from app.schemas.query import Query


SYSTEM_INSTRUCTION = """你是一个严谨的企业财务报告问答助手。
请仅依据提供的参考内容回答问题，不要编造参考内容中没有的信息。
如果参考内容不足以回答问题，请明确说明“参考内容不足以回答该问题”。
回答应使用中文，简洁、直接；涉及数字时保留原始单位，并在适当位置标注来源和页码。
"""


def build_prompt(query: Query | str, context: str) -> str:
	"""Build the instruction and grounded context sent to the LLM."""
	validated_query = query if isinstance(query, Query) else Query(text=query)
	return (
		f"{SYSTEM_INSTRUCTION}\n\n"
		f"参考内容:\n{context}\n\n"
		f"用户问题:\n{validated_query.text}\n\n"
		"请给出最终答案："
	)
