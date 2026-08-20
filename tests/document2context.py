from langchain_core.documents import Document


def documents_to_context(documents: list[Document]) -> list[str]:
	"""Extract and store the text content from retrieved documents."""
	return [document.page_content for document in documents]
