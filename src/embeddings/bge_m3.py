from typing import List

from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings

from typing import List

from FlagEmbedding import BGEM3FlagModel
from langchain_core.embeddings import Embeddings


class BGEM3Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 8192,
    ):
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
        )

        self.batch_size = batch_size
        self.max_length = max_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Socket 1 : Transform a list of documents into embeddings."""
        if not texts:
            return []
        
        output = self.model.encode_corpus(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        """Socket 2 : Transform a query into an embedding."""
        output = self.model.encode_queries(
            [text],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return output["dense_vecs"][0].tolist()
