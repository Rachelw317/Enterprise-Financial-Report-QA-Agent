from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str = Field(description="Raw PDF File Name.")
    page: int = Field(description="Page No.")
    document_year: int = Field(description="PDF Year")
    data_year: list[int] = Field(
        default_factory=list,
        description="Financial Data Year (May contain multiple years)"
    )


class Document(BaseModel):
    content: str = Field(description="Document Chunk Content")
    metadata: DocumentMetadata