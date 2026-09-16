from typing import Any, TypeVar

from pydantic import BaseModel

from app.graph.nodes._utils import invoke_structured, response_text


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def invoke_text(llm: Any, prompt: str) -> str:
	return response_text(llm.invoke(prompt))


def invoke_schema(llm: Any, schema: type[SchemaT], prompt: str) -> SchemaT:
	return invoke_structured(llm, schema, prompt)