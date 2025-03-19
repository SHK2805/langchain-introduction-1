from src.Constants import *
from src.Loaders.text_loader import TextToVectorStoreManager
from src.Services.query_manager_service import QueryManager
from src.Services.vector_store_manager_service import VectorStoreManager
from src.utils.common_utils import *

if __name__ == "__main__":
    # File paths and configurations
    txt_file_path = "pyramids.txt"
    index_file_path = get_fiass_vector_store_path()
    mapping_file_path =  get_fiass_vector_store_mapping_path()
    docstore_file = get_fiass_docfile_path()

    # Initialize manager
    manager = TextToVectorStoreManager(
        txt_file_path=txt_file_path,
        index_file_path=index_file_path,
        mapping_file_path=mapping_file_path,
        docstore_file=docstore_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Process text and manage vector store
    manager.process_text()
    manager.manage_vector_store()

    # Query the vector store
    query = "What is the significance of the Pyramids of Giza?"
    manager.query_vector_store(query)




