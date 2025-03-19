import os
from uuid import uuid4

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.Config.set_config import Config
from src.Constants import openai_embeddings_model_name
from src.utils.project_environment.envs import get_openai_llm_model_name
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

def get_file_extension(file_path:str):
    return Path(file_path).suffix

def get_document_map():
    return {
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".pdf": PyPDFLoader,
    }

def get_loader(file_path, mapping=None):
    if mapping is None:
        mapping = get_document_map()
    ext = get_file_extension(file_path)
    loader = mapping.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(file_path)


def get_template_prompt():
    # Define a prompt template for question answering
    system_prompt = """Answer the following question based ONLY on the provided context. Do not use any external information. Don't worry if you don't know the answer. Just say I don't know. Do not hallucinate or make up information. Only answer based on the provided context.:
    <context>
    {context}
    </context>"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    return prompt

def get_message_prompt():
    # Define a prompt template for question answering
    system_prompt = """Answer the following question based ONLY on the provided context. Do not use any external information. Don't worry if you don't know the answer. Do not hallucinate or make up information. Only answer based on the provided context.:
    <context>
    {context}
    </context>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return prompt

def get_combined_chain(llm_model_name = None, prompt = None):
    # prompt the question
    if prompt is None:
        prompt = get_message_prompt()
    # create the chain
    llm = get_openai_llm_model(llm_model_name)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return combine_docs_chain

def get_new_vector_store():
    """
    Create a new vectorstore
    :return: vectorstore
    """
    """
    The FAISS index requires the dimensionality of the vectors you'll store in it. This ensures that all vectors added to the index have a consistent size.
    "hello world" is passed to an embedding function, which converts the phrase into a numerical vector.
    The length (number of dimensions) of the embedding is calculated using len(...).
    This dimensionality is passed as an argument to faiss.IndexFlatL2, creating an index that can store and perform operations on vectors of the same size.
    """
    index = faiss.IndexFlatL2(len(get_openai_embeddings().embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=get_openai_embeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    return vector_store

def get_openai_embeddings(embeddings_model_name = None):
    if embeddings_model_name is None:
        embeddings_model_name = openai_embeddings_model_name
    return OpenAIEmbeddings(model=embeddings_model_name)


def get_openai_llm_model(openai_llm_model_name = None):
    if openai_llm_model_name is None:
        openai_llm_model_name = get_openai_llm_model_name()
    return ChatOpenAI(model=openai_llm_model_name)

def get_uuids(docs_len):
    return [str(uuid4()) for _ in range(docs_len) if docs_len > 0]

def get_fiass_vector_store_path():
    return os.getenv('FIASS_INDEX_FILE_PATH')

def get_fiass_vector_store_mapping_path():
    return os.getenv('FIASS_INDEX_MAPPING_FILE_PATH')

def get_fiass_docfile_path():
    return os.getenv('FIASS_DOC_FILE_PATH')



