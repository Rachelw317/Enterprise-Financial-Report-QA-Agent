from pathlib import Path

from langchain_chroma import Chroma
from src.embeddings.bge_m3 import BGEM3Embeddings


# Chroma vector database path
db_path = Path(__file__).parent.parent / "data" / "vector_db" / "chroma_db"

embedding_model = BGEM3Embeddings(
    batch_size=8
)

# Chroma Retriever instance
retriever = Chroma(
    collection_name="test_collection",
    persist_directory=str(db_path),
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 3})
