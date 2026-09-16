import unittest

from app.graph.graph import run_rag_graph
from app.schemas.document import Document, DocumentMetadata
from app.schemas.grade import GradeResult
from app.schemas.query import Query
from app.schemas.query_analyze import QueryAnalysis
from app.schemas.rewritten_query import RewrittenQuery
from app.schemas.validation import ValidationResult


class FakeStructuredLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        if self.schema is QueryAnalysis:
            return QueryAnalysis(
                query_type="comparison",
                metrics=["营业收入"],
                years=[2022, 2023],
                operations=["difference"],
                is_complex=True,
                retrieval_queries=["2022 年营业收入", "2023 年营业收入"],
            )
        if self.schema is RewrittenQuery:
            original = Query(text="比较 2022 和 2023 年营业收入")
            return RewrittenQuery(
                original_query=original,
                sub_queries=[
                    Query(text="2022 年营业收入"),
                    Query(text="2023 年营业收入"),
                ],
                reason="complex_query",
            )
        if self.schema is GradeResult:
            return GradeResult(relevant=True, feedback="证据完整")
        if self.schema is ValidationResult:
            return ValidationResult(valid=True, feedback="答案有证据支持")
        raise AssertionError(self.schema)


class FakeLLM:
    def with_structured_output(self, schema):
        return FakeStructuredLLM(schema)

    def invoke(self, prompt):
        return "2023 年营业收入高于 2022 年。"


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query.text)
        return [
            Document(
                content=f"证据：{query.text}",
                metadata=DocumentMetadata(
                    source="test.pdf",
                    page=1,
                    document_year=2023,
                ),
            )
        ]


class GraphTests(unittest.TestCase):
    def test_complex_query_retrieves_each_sub_query(self):
        retriever = FakeRetriever()
        state = run_rag_graph(
            "比较 2022 和 2023 年营业收入",
            FakeLLM(),
            retriever,
        )

        self.assertEqual(
            retriever.queries,
            ["2022 年营业收入", "2023 年营业收入"],
        )
        self.assertEqual(len(state.context), 2)
        self.assertEqual(state.answer, "2023 年营业收入高于 2022 年。")
        self.assertTrue(state.validation_result)


if __name__ == "__main__":
    unittest.main()