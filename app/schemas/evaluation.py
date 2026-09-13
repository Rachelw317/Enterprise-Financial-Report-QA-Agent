from typing import Literal
from pydantic import BaseModel, Field


class RequiredEvidence(BaseModel):
    source: str = Field(
        description="证据所在的原始 PDF 文件名"
    )

    page: int = Field(
        description="证据所在的 PDF 页码"
    )

    data_year: list[int] = Field(
        default_factory=list,
        description="该证据涉及的财务数据年份"
    )


class EvaluationCase(BaseModel):
    id: str = Field(
        description="Evaluation question 的唯一 ID"
    )

    query: str = Field(
        min_length=1,
        description="用户提出的问题"
    )

    type: Literal["fact", "comparison", "trend"] = Field(
        description="问题类型"
    )

    expected_answer: str = Field(
        min_length=1,
        description="人工编写的标准答案"
    )

    required_evidence: list[RequiredEvidence] = Field(
        default_factory=list,
        description="回答该问题所必需的证据"
    )


class EvaluationResult(BaseModel):
    id: str = Field(
        description="对应 EvaluationCase 的 ID"
    )

    query: str = Field(
        description="实际执行的 Query"
    )

    retrieved_evidence: list[RequiredEvidence] = Field(
        default_factory=list,
        description="系统实际检索到的相关证据"
    )

    generated_answer: str = Field(
        description="RAG 系统实际生成的答案"
    )

    retrieval_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="检索证据召回率；未执行评分时为空"
    )

    answer_score: Literal[0, 1, 2] | None = Field(
        default=None,
        description="答案正确性评分；未执行评分时为空"
    )

    groundedness_score: Literal[0, 1, 2] | None = Field(
        default=None,
        description="答案证据支持度评分；未执行评分时为空"
    )

    comment: str = Field(
        default="",
        description="人工或 LLM Judge 对本题的评价说明"
    )