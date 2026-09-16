from typing import Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    query_type: Literal[
        "fact",
        "comparison",
        "trend",
    ] = Field(
        description="问题类型"
    )

    metrics: list[str] = Field(
        default_factory=list,
        description="用户希望查询的指标，例如营业收入、净利润、总资产"
    )

    years: list[int] = Field(
        default_factory=list,
        description="问题涉及的年份"
    )

    operations: list[
        Literal[
            "lookup",
            "difference",
            "growth_rate",
            "trend",
            "cagr",
            "explanation",
        ]
    ] = Field(
        default_factory=list,
        description="回答问题时需要执行的操作"
    )

    is_complex: bool = Field(
        default=False,
        description="是否需要拆分为多个子问题分别检索"
    )

    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="后续 Retriever 可以使用的检索 Query"
    )

