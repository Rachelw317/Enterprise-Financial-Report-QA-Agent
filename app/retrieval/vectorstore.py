import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

from app.retrieval.embedding import create_embeddings
from app.schemas.document import Document, DocumentMetadata


class VectorStoreBackend(Protocol):
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, str | int]],
        ids: Sequence[str],
    ) -> None: ...

    def similarity_search(self, query: str, k: int) -> list[LangChainDocument]: ...

    def count(self) -> int: ...


class ChromaBackend:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self.store = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, str | int]],
        ids: Sequence[str],
    ) -> None:
        self.store.add_texts(
            texts=list(texts),
            metadatas=list(metadatas),
            ids=list(ids),
        )

    def similarity_search(self, query: str, k: int) -> list[LangChainDocument]:
        return self.store.similarity_search(query, k=k)

    def count(self) -> int:
        return self.store._collection.count()


BackendFactory = Callable[..., VectorStoreBackend]
_BACKEND_FACTORIES: dict[str, BackendFactory] = {"chroma": ChromaBackend}


def register_vector_store_backend(name: str, factory: BackendFactory) -> None:
    """Register a database backend for use by ``VectorStore``."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Vector store backend name must not be empty")
    _BACKEND_FACTORIES[normalized_name] = factory


class VectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "financial_reports",
        embeddings: Embeddings | None = None,
        embedding_model: str = "bgem3",
        backend: VectorStoreBackend | None = None,
        database: str = "chroma",
        **embedding_kwargs: Any,
    ) -> None:
        self.embeddings = embeddings or create_embeddings(
            embedding_model,
            **embedding_kwargs,
        )
        self.backend = backend or self._create_backend(
            database=database,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embeddings=self.embeddings,
        )

    @staticmethod
    def _create_backend(database: str, **kwargs: Any) -> VectorStoreBackend:
        normalized_name = database.strip().lower()
        try:
            factory = _BACKEND_FACTORIES[normalized_name]
        except KeyError as error:
            supported_backends = ", ".join(sorted(_BACKEND_FACTORIES))
            raise ValueError(
                f"Unsupported vector store backend {database!r}. "
                f"Supported backends: {supported_backends}"
            ) from error
        return factory(**kwargs)

    @staticmethod
    def _metadata_to_dict(document: Document) -> dict[str, str | int]:
        return {
            "source": document.metadata.source,
            "page": document.metadata.page,
            "document_year": document.metadata.document_year,
            "data_year": json.dumps(document.metadata.data_year),
        }

    @staticmethod
    def _to_document(content: str, metadata: dict[str, object]) -> Document:
        data_year_value = metadata.get("data_year", "[]")
        data_year = (
            json.loads(data_year_value)
            if isinstance(data_year_value, str)
            else data_year_value
        )
        return Document(
            content=content,
            metadata=DocumentMetadata(
                source=str(metadata["source"]),
                page=int(metadata["page"]),
                document_year=int(metadata["document_year"]),
                data_year=[int(year) for year in data_year],
            ),
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not documents:
            return []
        if ids is not None and len(ids) != len(documents):
            raise ValueError("The number of ids must match the number of documents")

        document_ids = list(ids) if ids is not None else [str(uuid4()) for _ in documents]
        self.backend.add_texts(
            texts=[document.content for document in documents],
            metadatas=[self._metadata_to_dict(document) for document in documents],
            ids=document_ids,
        )
        return document_ids

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if k < 1:
            raise ValueError("k must be greater than 0")
        results = self.backend.similarity_search(query, k=k)
        return [self._to_document(result.page_content, result.metadata) for result in results]

    def count(self) -> int:
        return self.backend.count()
