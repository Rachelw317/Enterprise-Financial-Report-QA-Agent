from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.document import Document


def split_documents(
	documents: Iterable[Document],
	chunk_size: int = 1000,
	chunk_overlap: int = 200,
) -> list[Document]:
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	chunks: list[Document] = []

	for document in documents:
		for content in text_splitter.split_text(document.content):
			chunks.append(
				Document(
					content=content,
					metadata=document.metadata.model_copy(deep=True),
				)
			)

	return chunks
