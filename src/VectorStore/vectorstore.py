import faiss
import os
import pickle
import numpy as np
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from src.VectorStore.custom_docstore import CustomDocstore
from src.utils.common_utils import get_openai_embeddings


class FaissVectorStore:
    def __init__(self, index_file=None, mapping_file=None, docstore_file=None):
        """
        Initialize the FAISS vector store.

        Args:
            index_file (str, optional): Path to save/load the FAISS index.
            mapping_file (str, optional): Path to save/load the index_to_docstore_id mapping.
            docstore_file (str, optional): Path to save/load the docstore.
        """
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.docstore_file = docstore_file
        self.vector_store = None

    def get_or_create_vector_store(self):
        if self.vector_store is None:
            if self.index_file and os.path.exists(self.index_file):
                index = faiss.read_index(self.index_file)
                print("Loaded FAISS index from file.")

                # Load index_to_docstore_id mapping
                if self.mapping_file and os.path.exists(self.mapping_file):
                    with open(self.mapping_file, "rb") as f:
                        index_to_docstore_id = pickle.load(f)
                        print("Loaded index_to_docstore_id mapping from file.")
                else:
                    index_to_docstore_id = {}
                    print("No mapping file found. Using an empty mapping.")

                # Load docstore
                if self.docstore_file and os.path.exists(self.docstore_file):
                    with open(self.docstore_file, "rb") as f:
                        docstore_data = pickle.load(f)
                        docstore = CustomDocstore(docstore_data)
                        print("Loaded docstore from file.")
                else:
                    docstore = CustomDocstore()  # Initialize an empty CustomDocstore
                    print("No docstore file found. Using a new CustomDocstore.")
            else:
                # Initialize new components
                dimension = len(get_openai_embeddings().embed_query("hello world"))
                index = faiss.IndexFlatL2(dimension)
                index_to_docstore_id = {}
                docstore = CustomDocstore()  # Use CustomDocstore
                print("Created a new FAISS index, mapping, and CustomDocstore.")

            self.vector_store = FAISS(
                embedding_function=get_openai_embeddings(),
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )

        return self.vector_store

    def save_index(self):
        """
        Saves the FAISS index, index_to_docstore_id, and docstore to their respective files.
        """
        if self.index_file and self.vector_store:
            # Save the FAISS index
            faiss.write_index(self.vector_store.index, self.index_file)
            print(f"FAISS index saved to {self.index_file}.")

        if self.mapping_file and self.vector_store:
            # Save the index_to_docstore_id mapping
            with open(self.mapping_file, "wb") as f:
                pickle.dump(self.vector_store.index_to_docstore_id, f)
                print(f"Mapping saved to {self.mapping_file}.")

        if isinstance(self.vector_store.docstore, CustomDocstore):
            # Serialize CustomDocstore and save
            with open(self.docstore_file, "wb") as f:
                pickle.dump(self.vector_store.docstore.store, f)  # Save internal dictionary directly
                print(f"Docstore saved to {self.docstore_file}.")

    def add_documents(self, documents, ids):
        """
        Add documents to the FAISS index and update mappings.

        Args:
            documents (list): List of strings (document content).
            ids (list): List of document IDs corresponding to the documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not created. Call get_or_create_vector_store() first.")

        # Update the docstore with new documents
        if isinstance(self.vector_store.docstore, CustomDocstore):
            # Use CustomDocstore's `add_documents` method
            self.vector_store.docstore.add_documents({doc_id: doc for doc, doc_id in zip(documents, ids)})
        elif isinstance(self.vector_store.docstore, dict):
            # Handle docstore as dictionary
            for doc, doc_id in zip(documents, ids):
                self.vector_store.docstore[doc_id] = doc
        else:
            raise TypeError("Unsupported docstore type. Must be CustomDocstore or dictionary.")

        # Update the index_to_docstore_id mapping
        for doc_id in ids:
            self.vector_store.index_to_docstore_id[len(self.vector_store.index_to_docstore_id)] = doc_id

        # Embed documents into vectors and add to the FAISS index
        vectors = [self.vector_store.embedding_function.embed_query(doc) for doc in documents]
        self.vector_store.index.add(np.array(vectors).astype("float32"))

        print(f"Added {len(documents)} documents to the FAISS vector store.")







