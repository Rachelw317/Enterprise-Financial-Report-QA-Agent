from app.graph.state import RAGState
from app.schemas.validation import ValidationResult

from ._llm import invoke_schema


def build_validation_prompt(state: RAGState) -> str:
	query = state.query.text if state.query else ""
	references = "\n".join(document.content for document in state.context) or "无"
	return f"""请验证下面的财务问答结果，并严格返回 ValidationResult 结构。

用户问题：{query}
参考内容：
{references}

答案：
{state.answer or '无'}

检查答案是否真正回答了问题、是否被参考内容支持、是否出现编造数字。
"""


def validate_answer(llm, state: RAGState) -> ValidationResult:
	return invoke_schema(llm, ValidationResult, build_validation_prompt(state))