from app.graph.state import RAGState
from app.schemas.validation import ValidationResult
from app.tools.answer_validator import validate_answer


def create_validate_node(llm=None):
    def validate_node(state: RAGState) -> dict:
        if llm is None:
            result = ValidationResult(
                valid=bool(state.answer and state.context),
                feedback="答案和参考内容存在" if state.answer and state.context else "答案或参考内容为空",
            )
        else:
            result = validate_answer(llm, state)
        return {
            "validation_result": result.valid,
            "validation_feedback": result.feedback,
            "retry_count": state.retry_count + (0 if result.valid else 1),
        }

    return validate_node