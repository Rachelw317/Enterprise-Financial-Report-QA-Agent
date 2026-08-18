from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.bge_m3 import BGEM3Embeddings
from tests.test_splitter import chunks


db_path = Path(__file__).parent.parent / "data" / "vector_db" / "chroma_db"

embedding_model = BGEM3Embeddings(
    batch_size=8
)

# Test only first 16 chunks for embedding and vector store creation
vector_store = Chroma.from_texts(
    texts=chunks[:16],
    embedding=embedding_model, # Should be a type of Embeddings, not a list of calculated vectors. 
    collection_name="test_collection",
    persist_directory=str(db_path),
)

print(
    f"Number of embeddings in collection: "
    f"{vector_store._collection.count()}"
)