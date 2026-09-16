from typing import Any

from app.graph.nodes.generate import create_generate_node
from app.graph.nodes.grade import create_grade_node
from app.graph.nodes.query_analyze import create_query_analyze_node
from app.graph.nodes.query_rewrite import create_query_rewrite_node
from app.graph.nodes.retrieve import create_retrieve_node
from app.graph.nodes.validate import create_validate_node
from app.graph.state import RAGState
from app.schemas.query import Query


def route_after_analysis(state: RAGState) -> str:
	analysis = state.query_analysis
	if analysis is None:
		raise ValueError("Query analysis is required before routing")
	if analysis.is_complex or len(analysis.retrieval_queries) > 1:
		return "rewrite"
	return "retrieve"


def route_after_retrieve(state: RAGState) -> str:
	if (
		state.rewritten_query is not None
		and state.sub_query_index < len(state.rewritten_query.sub_queries)
	):
		return "retrieve"
	return "grade"


def route_after_grade(state: RAGState, max_retries: int = 2) -> str:
	if state.relevant_result:
		return "generate"
	if state.retry_count < max_retries:
		return "rewrite"
	return "generate"


def route_after_validate(state: RAGState, max_retries: int = 2) -> str:
	if state.validation_result:
		return "end"
	if state.retry_count < max_retries:
		return "generate"
	return "end"


def create_rag_graph(
	llm: Any,
	retriever: Any,
	*,
	max_retries: int = 2,
):
	if max_retries < 0:
		raise ValueError("max_retries must be greater than or equal to 0")
	try:
		from langgraph.graph import END, START, StateGraph
	except ImportError as error:
		raise RuntimeError(
			"LangGraph is required to build the RAG graph. "
			"Install dependencies from requirements.txt."
		) from error

	graph = StateGraph(RAGState)
	graph.add_node("query_analyze", create_query_analyze_node(llm))
	graph.add_node("query_rewrite", create_query_rewrite_node(llm))
	graph.add_node("retrieve", create_retrieve_node(retriever))
	graph.add_node("grade", create_grade_node(llm))
	graph.add_node("generate", create_generate_node(llm))
	graph.add_node("validate", create_validate_node(llm))

	graph.add_edge(START, "query_analyze")
	graph.add_conditional_edges(
		"query_analyze",
		route_after_analysis,
		{"rewrite": "query_rewrite", "retrieve": "retrieve"},
	)
	graph.add_edge("query_rewrite", "retrieve")
	graph.add_conditional_edges(
		"retrieve",
		route_after_retrieve,
		{"retrieve": "retrieve", "grade": "grade"},
	)
	graph.add_conditional_edges(
		"grade",
		lambda state: route_after_grade(state, max_retries),
		{"rewrite": "query_rewrite", "generate": "generate"},
	)
	graph.add_edge("generate", "validate")
	graph.add_conditional_edges(
		"validate",
		lambda state: route_after_validate(state, max_retries),
		{"generate": "generate", "end": END},
	)
	return graph.compile()


def run_rag_graph(
	query: Query | str,
	llm: Any,
	retriever: Any,
	*,
	max_retries: int = 2,
) -> RAGState:
	graph = create_rag_graph(llm, retriever, max_retries=max_retries)
	initial_state = RAGState(
		query=query if isinstance(query, Query) else Query(text=query)
	)
	result = graph.invoke(initial_state)
	return result if isinstance(result, RAGState) else RAGState.model_validate(result)
