from pathlib import Path
from typing import Final

from app.retrieval.vectorstore import VectorStore
from app.schemas.document import Document
from app.schemas.query import Query


class Retriever:
	TOP_K: Final = 5

	def __init__(self, vectorstore: VectorStore) -> None:
		self.vectorstore = vectorstore

	def retrieve(self, query: Query | str) -> list[Document]:
		validated_query = query if isinstance(query, Query) else Query(text=query)
		return self.vectorstore.similarity_search(
			validated_query.text,
			k=self.TOP_K,
		)


def create_retriever(
	vectorstore_directory: Path,
	collection_name: str = "financial_reports",
	embedding_model: str = "bgem3",
) -> Retriever:
	vectorstore = VectorStore(
		persist_directory=vectorstore_directory,
		collection_name=collection_name,
		embedding_model=embedding_model,
	)
	return Retriever(vectorstore)
