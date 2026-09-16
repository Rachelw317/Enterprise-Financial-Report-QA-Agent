from app.graph.state import RAGState
from app.tools.query_rewriter import rewrite_query


def create_query_rewrite_node(llm):
    def query_rewrite_node(state: RAGState) -> dict:
        rewritten = rewrite_query(llm, state)
        retry_count = state.retry_count
        if rewritten.reason == "irrelevant_documents":
            retry_count += 1
        return {
            "rewritten_query": rewritten,
            "rewrite_history": [*state.rewrite_history, rewritten],
            "active_query": None,
            "sub_query_index": 0,
            "context": [],
            "relevant_result": None,
            "grade_feedback": None,
            "retry_count": retry_count,
        }

    return query_rewrite_node