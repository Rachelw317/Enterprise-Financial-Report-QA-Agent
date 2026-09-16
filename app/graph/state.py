from pydantic import BaseModel, Field

from app.schemas.query import Query
from app.schemas.document import Document
from app.schemas.query_analyze import QueryAnalysis
from app.schemas.rewritten_query import RewrittenQuery


class RAGState(BaseModel):
    query: Query | None = None
    query_analysis: QueryAnalysis | None = None

    active_query: Query | None = Field(
        default=None,
        description="当前正在执行 RAG 检索的 Query"
    )
    sub_query_index: int = Field(
        default=0,
        ge=0,
        description="当前要处理的子 Query 在重写结果中的索引"
    )

    context: list[Document] = Field(
        default_factory=list,
        description="当前用于生成答案的去重后文档集合"
    )
    retrieval_results: dict[str, list[Document]] = Field(
        default_factory=dict,
        description="按 Query ID 保存每次检索得到的文档，便于追踪子 Query 的证据"
    )

    rewritten_query: RewrittenQuery | None = None
    rewrite_history: list[RewrittenQuery] = Field(
        default_factory=list,
        description="历次 Query 重写结果，用于记录重试过程"
    )

    relevant_result: bool | None = None
    grade_feedback: str | None = Field(
        default=None,
        description="文档相关性判断的原因或改写建议"
    )
    
    answer: str | None = None

    retry_count: int = Field(
        default=0,
        ge=0,
        description="检索或生成失败后的重试次数"
    )

    validation_result: bool | None = None
    validation_feedback: str | None = Field(
        default=None,
        description="答案验证失败的原因或修正建议"
    )