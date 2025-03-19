from langchain_community.docstore import InMemoryDocstore

class CustomDocstore(InMemoryDocstore):
    def __init__(self, documents=None):
        """
        Initialize the CustomDocstore.

        Args:
            documents (dict, optional): Initial documents for the docstore.
        """
        super().__init__(documents or {})

    def add_documents(self, documents):
        """Add multiple documents to the docstore."""
        for doc_id, doc_content in documents.items():
            self._dict[doc_id] = doc_content  # Ensure `_dict` exists in `InMemoryDocstore`

    def get_document_ids(self):
        """Retrieve all document IDs."""
        return list(self._dict.keys())

    def get_documents(self):
        """Retrieve all documents as a dictionary."""
        return self._dict  # Expose internal dictionary for external access
