from app.graph.state import RAGState
from app.schemas.query import Query
from app.tools.query_analyzer import analyze_query


def create_query_analyze_node(llm):
    def query_analyze_node(state: RAGState) -> dict:
        if state.query is None:
            raise ValueError("RAGState.query is required before query analysis")
        analysis = analyze_query(llm, state.query)
        active_query = (
            Query(text=analysis.retrieval_queries[0])
            if analysis.retrieval_queries
            else state.query
        )
        return {
            "query_analysis": analysis,
            "active_query": active_query,
            "sub_query_index": 0,
        }

    return query_analyze_node