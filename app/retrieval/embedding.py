from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings


EmbeddingFactory = Callable[..., Embeddings]


class BGEM3Embeddings(Embeddings):
	"""LangChain embedding adapter for BAAI/bge-m3 dense embeddings."""

	DIMENSION = 1024

	def __init__(
		self,
		model_name: str = "BAAI/bge-m3",
		batch_size: int = 8,
		max_length: int = 8192,
		**model_kwargs: Any,
	) -> None:
		if batch_size < 1:
			raise ValueError("batch_size must be greater than 0")
		if max_length < 1:
			raise ValueError("max_length must be greater than 0")

		self.model_name = model_name
		self.batch_size = batch_size
		self.max_length = max_length
		self.model_kwargs = model_kwargs
		self._model: Any | None = None

	@property
	def model(self) -> Any:
		if self._model is None:
			from FlagEmbedding import BGEM3FlagModel

			self._model = BGEM3FlagModel(self.model_name, **self.model_kwargs)
		return self._model

	def _encode(self, texts: Sequence[str]) -> list[list[float]]:
		if not texts:
			return []

		result = self.model.encode(
			list(texts),
			batch_size=self.batch_size,
			max_length=self.max_length,
		)
		vectors = result["dense_vecs"]
		if len(vectors) != len(texts):
			raise ValueError("BGE-M3 returned a different number of vectors than texts")

		result_vectors = [
			vector.tolist() if hasattr(vector, "tolist") else list(vector)
			for vector in vectors
		]
		if any(len(vector) != self.DIMENSION for vector in result_vectors):
			raise ValueError(
				f"Expected {self.DIMENSION}-dimensional vectors from BGE-M3"
			)
		return result_vectors

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return self._encode(texts)

	def embed_query(self, text: str) -> list[float]:
		vectors = self._encode([text])
		return vectors[0]


_EMBEDDING_FACTORIES: dict[str, EmbeddingFactory] = {
	"bgem3": BGEM3Embeddings,
}


def register_embedding_model(name: str, factory: EmbeddingFactory) -> None:
	"""Register an embedding implementation for use by ``create_embeddings``."""
	normalized_name = name.strip().lower()
	if not normalized_name:
		raise ValueError("Embedding model name must not be empty")
	_EMBEDDING_FACTORIES[normalized_name] = factory


def create_embeddings(model_name: str = "bgem3", **kwargs: Any) -> Embeddings:
	"""Create the configured embedding model through a switchable registry."""
	normalized_name = model_name.strip().lower()
	try:
		factory = _EMBEDDING_FACTORIES[normalized_name]
	except KeyError as error:
		supported_models = ", ".join(sorted(_EMBEDDING_FACTORIES))
		raise ValueError(
			f"Unsupported embedding model {model_name!r}. "
			f"Supported models: {supported_models}"
		) from error
	return factory(**kwargs)
