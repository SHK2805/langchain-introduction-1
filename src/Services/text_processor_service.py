import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.Constants import *
from src.utils.common_utils import get_loader


class TextProcessor:
    def __init__(self, txt_file_path):
        self.txt_file_path = txt_file_path
        self.docs = None
        self.split_docs = None

    def load_and_split_text(self, size=chunk_size, overlap=chunk_overlap):
        # Load the text file
        if not os.path.exists(self.txt_file_path):
            raise FileNotFoundError(f"File not found: {self.txt_file_path}")

        loader = get_loader(self.txt_file_path)
        self.docs = loader.load()

        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        self.split_docs = splitter.split_documents(self.docs)

        print(f"Text loaded and split into {len(self.split_docs)} chunks.")
        return self.split_docs