from app.graph.state import RAGState
from app.schemas.query import Query

from ._utils import merge_documents


def _next_query(state: RAGState) -> Query:
    if state.rewritten_query is not None:
        queries = state.rewritten_query.sub_queries
        if state.sub_query_index >= len(queries):
            raise ValueError("No remaining rewritten query to retrieve")
        return queries[state.sub_query_index]
    if state.query is None:
        raise ValueError("RAGState.query is required before retrieval")
    return state.query


def create_retrieve_node(retriever):
    def retrieve_node(state: RAGState) -> dict:
        query = _next_query(state) if state.rewritten_query else state.active_query or _next_query(state)
        documents = list(retriever.retrieve(query))
        retrieval_results = dict(state.retrieval_results)
        retrieval_results[str(query.query_id)] = documents
        next_index = state.sub_query_index + 1 if state.rewritten_query else 0
        return {
            "active_query": query,
            "sub_query_index": next_index,
            "retrieval_results": retrieval_results,
            "context": merge_documents(state.context, documents),
        }

    return retrieve_node