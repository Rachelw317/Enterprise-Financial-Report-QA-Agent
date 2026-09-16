from app.graph.state import RAGState
from app.schemas.grade import GradeResult
from app.tools.document_grader import grade_documents


def create_grade_node(llm=None):
    def grade_node(state: RAGState) -> dict:
        if llm is None:
            result = GradeResult(
                relevant=bool(state.context),
                feedback="存在检索文档" if state.context else "未检索到文档",
            )
        else:
            result = grade_documents(llm, state)
        return {
            "relevant_result": result.relevant,
            "grade_feedback": result.feedback,
        }

    return grade_node