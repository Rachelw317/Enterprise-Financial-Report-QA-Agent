from src.embeddings.bge_m3 import BGEM3Embeddings

from tests.test_splitter import chunks

embeddings = []

embedding_model = BGEM3Embeddings()

# Let the embedding model process the chunks in batch itself, 
# so we can pass the loop to the model and let it handle batching internally.
embeddings = embedding_model.embed_documents(chunks)


# print first 16 embeddings for testing
for i, embedding in enumerate(embeddings[:16]):
    print(f"Embedding {i + 1}: Length: {len(embedding)}")