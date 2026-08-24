from pydantic import BaseModel, Field

from .document import Document


class Context(BaseModel):
    documents: list[Document] = Field(
        default_factory=list,
        description="用于回答当前 Query 的检索结果"
    )