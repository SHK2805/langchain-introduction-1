from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.Config.set_config import Config
from src.Constants import openai_embeddings_model_name

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

DOCUMENT_MAP = {
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
        mapping = DOCUMENT_MAP
    ext = file_path.suffix
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
    system_prompt = """Answer the following question based ONLY on the provided context. Do not use any external information. Don't worry if you don't know the answer. Just say I don't know. Do not hallucinate or make up information. Only answer based on the provided context.:
    <context>
    {context}
    </context>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return prompt

def get_combined_chain(llm_model_name):
    # prompt the question
    prompt = get_message_prompt()
    # create the chain
    llm = ChatOpenAI(model=llm_model_name)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return combine_docs_chain

def get_new_vector_store():
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )


