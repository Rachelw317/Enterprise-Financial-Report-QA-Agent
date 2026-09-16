from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    valid: bool = Field(
        description="答案是否有参考内容支持并且正确回答了用户问题"
    )
    feedback: str = Field(
        default="",
        description="答案验证的理由，以及失败时的修正建议"
    )