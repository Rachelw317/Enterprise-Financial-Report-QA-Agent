from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    relevant: bool = Field(
        description="当前检索结果是否足以支持回答用户问题"
    )
    feedback: str = Field(
        default="",
        description="相关性判断的理由，以及无关时可用于改写 Query 的建议"
    )