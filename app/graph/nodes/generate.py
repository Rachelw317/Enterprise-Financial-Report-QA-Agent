from app.generation.context import format_context
from app.generation.prompt import build_prompt
from app.graph.state import RAGState

from ._utils import response_text


def create_generate_node(llm):
    def generate_node(state: RAGState) -> dict:
        if state.query is None:
            raise ValueError("RAGState.query is required before generation")
        prompt = build_prompt(state.query, format_context(state.context))
        if state.validation_feedback:
            prompt += f"\n\n上一次答案验证反馈：{state.validation_feedback}\n请据此修正答案。"
        return {"answer": response_text(llm.invoke(prompt))}

    return generate_node