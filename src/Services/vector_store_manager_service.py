from src.VectorStore.vectorstore import FaissVectorStore


class VectorStoreManager:
    def __init__(self, index_file, mapping_file, docstore_file):
        """
        Initialize the manager with file paths for the FAISS index, mappings, and docstore.

        Args:
            index_file (str): Path to save/load the FAISS index.
            mapping_file (str): Path to save/load the index_to_docstore_id mapping.
            docstore_file (str): Path to save/load the docstore.
        """
        self.faiss_store = FaissVectorStore(
            index_file=index_file,
            mapping_file=mapping_file,
            docstore_file=docstore_file
        )
        self.vector_store = self.faiss_store.get_or_create_vector_store()

    def add_documents_to_store(self, documents, ids):
        """
        Add documents and their IDs to the vector store.

        Args:
            documents (list): List of Document objects to add.
            ids (list): List of document IDs corresponding to the documents.
        """
        # Extract text content from Document objects
        text_content = [doc.page_content for doc in documents]

        # Use FaissVectorStore's add_documents method
        self.faiss_store.add_documents(documents=text_content, ids=ids)

        # Save the index, mappings, and docstore
        self.faiss_store.save_index()
        print("Documents added and index saved successfully!")

    def get_retriever(self):
        """
        Returns a retriever for querying the vector store.

        Returns:
            Retriever: The vector store retriever.
        """
        return self.vector_store.as_retriever()
