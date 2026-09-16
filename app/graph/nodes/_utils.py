import json
from typing import Any, TypeVar

from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


def invoke_structured(
    llm: Any,
    schema: type[SchemaT],
    prompt: str,
) -> SchemaT:
    structured_llm = (
        llm.with_structured_output(schema)
        if hasattr(llm, "with_structured_output")
        else llm
    )
    response = structured_llm.invoke(prompt)
    if isinstance(response, schema):
        return response
    if isinstance(response, BaseModel):
        return schema.model_validate(response.model_dump())
    if isinstance(response, dict):
        return schema.model_validate(response)

    content = response_text(response)
    try:
        return schema.model_validate_json(content)
    except ValueError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM response is not valid {schema.__name__} JSON")
        return schema.model_validate(json.loads(content[start : end + 1]))


def document_key(document: Any) -> tuple[Any, ...]:
    metadata = document.metadata
    return (
        metadata.source,
        metadata.page,
        metadata.document_year,
        tuple(metadata.data_year),
        document.content,
    )


def merge_documents(existing: list[Any], incoming: list[Any]) -> list[Any]:
    merged = list(existing)
    known = {document_key(document) for document in merged}
    for document in incoming:
        key = document_key(document)
        if key not in known:
            merged.append(document)
            known.add(key)
    return merged