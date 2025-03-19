from src.Services.text_processor_service import TextProcessor
from src.Services.vector_store_manager_service import VectorStoreManager
from src.utils.common_utils import get_uuids


class TextToVectorStoreManager:
    def __init__(self, txt_file_path, index_file_path, mapping_file_path, docstore_file, chunk_size, chunk_overlap):
        """
        Initialize the manager with file paths, chunk size, and overlap.

        Args:
            txt_file_path (str): Path to the text file.
            index_file_path (str): Path to save/load the FAISS index.
            mapping_file_path (str): Path to save/load the index_to_docstore_id mapping.
            docstore_file (str): Path to save/load the docstore.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between text chunks.
        """
        self.txt_file_path = txt_file_path
        self.index_file_path = index_file_path
        self.mapping_file_path = mapping_file_path
        self.docstore_file = docstore_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_docs = None
        self.uuids = None
        self.vector_store_manager = None

    def process_text(self):
        """
        Load the text file, split it into chunks, and generate unique identifiers.
        """
        try:
            # Load the text file
            text_processor = TextProcessor(self.txt_file_path)
            self.split_docs = text_processor.load_and_split_text(size=self.chunk_size, overlap=self.chunk_overlap)

            if not self.split_docs:
                raise ValueError("No chunks were generated. Ensure the text file is not empty or misconfigured.")

            # Generate unique identifiers
            self.uuids = get_uuids(len(self.split_docs))
            print(f"Text processed into {len(self.split_docs)} chunks.")
        except Exception as e:
            raise RuntimeError(f"Failed to process text: {e}")

    def manage_vector_store(self):
        """
        Create or load the vector store and add documents.
        """
        if self.split_docs is None or self.uuids is None:
            raise ValueError("Text must be processed before managing the vector store.")

        # Debug: Print number of chunks and their IDs
        print(f"Preparing to add {len(self.split_docs)} chunks to the vector store.")
        print(f"Generated UUIDs: {self.uuids[:5]}...")  # Print a sample of the UUIDs

        # Manage vector store
        self.vector_store_manager = VectorStoreManager(
            index_file=self.index_file_path,
            mapping_file=self.mapping_file_path,
            docstore_file=self.docstore_file
        )
        self.vector_store_manager.add_documents_to_store(documents=self.split_docs, ids=self.uuids)
        print("Vector store updated successfully!")

    def query_vector_store(self, query):
        """
        Query the vector store.

        Args:
            query (str): The search query.

        Returns:
            list: Results from the vector store retriever.
        """
        if self.vector_store_manager is None:
            raise ValueError("Vector store has not been initialized. Call manage_vector_store first.")

        retriever = self.vector_store_manager.get_retriever()
        results = retriever.get_relevant_documents(query)
        print(f"Query results for '{query}':")
        for result in results:
            print(result)
        return results

