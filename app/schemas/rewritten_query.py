from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.query import Query


class RewrittenQuery(BaseModel):
	original_query: Query = Field(
		description="触发本次重写的用户原始问题"
	)

	sub_queries: list[Query] = Field(
		min_length=1,
		description="拆分后逐个用于 RAG 检索的小问题"
	)

	reason: Literal[
		"complex_query",
		"irrelevant_documents",
	] = Field(
		description="触发 Query 重写的原因"
	)
