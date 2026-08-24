from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class Query(BaseModel):
    query_id: UUID = Field(
        default_factory=uuid4,
        description="当前 Query 的唯一标识"
    )
    text: str = Field(
        min_length=1,
        description="用户输入的问题"
    )