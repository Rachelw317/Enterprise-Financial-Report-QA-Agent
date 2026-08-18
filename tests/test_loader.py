from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

# Resolve the path to the test PDF file
test_file_path = Path(__file__).parent.parent / "data" / "raw"

# Initialize the Document List
document_list = []

# Iterate through the test file path and load PDF documents
for pdf in test_file_path.iterdir():
    if pdf.suffix == ".pdf":
        loader = PyMuPDFLoader(str(pdf))
        document_list.extend(loader.load())


# Print the test results.

# for doc in document_list:
#     print(f"Document: {doc.metadata['source']}, Content Length: {len(doc.page_content)}")



