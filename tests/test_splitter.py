from langchain_text_splitters import RecursiveCharacterTextSplitter
from tests.test_loader import document_list


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in document_list:
    chunks.extend(splitter.split_text(doc.page_content))

# Print the test results.

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}: Length: {len(chunk)}")
    
