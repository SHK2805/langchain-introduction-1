from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.Config.set_config import Config
from src.Constants import openai_embeddings_model_name
from src.utils.project_environment.envs import get_openai_llm_model_name

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class WebPageQA:
    def __init__(self, url = "https://apnews.com/article/trump-putin-call-ceasefire-russia-ukraine-zelenskyy-0d2ca5b69761082979dd9836932ae84f",
                 llm_model_name = get_openai_llm_model_name(),
                 embeddings_model_name = openai_embeddings_model_name):
        self.url = url
        self.llm_model_name = llm_model_name
        self.embeddings_model_name = embeddings_model_name

    def init(self):
        # read the text from the webpage
        loader = WebBaseLoader(self.url)
        docs = loader.load()
        # split the text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        # get the embeddings for each chunk
        embeddings = OpenAIEmbeddings(model=self.embeddings_model_name)
        # create the vector store
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()
        return retriever

    def get_prompt(self):
        # Define a prompt template for question answering
        system_prompt = """Answer the following question based ONLY on the provided context:
        <context>
        {context}
        </context>"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return prompt

    def ask(self, retriever, question):
        # prompt the question
        prompt = self.get_prompt()
        # create the chain
        llm = ChatOpenAI(model=self.llm_model_name)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        # invoke the chain with the query
        response = retrieval_chain.invoke({"input": question})
        return response['answer']

    def run(self):
        retriever = self.init()
        answer = self.ask(retriever, "When is Trump holding talk with Putin?")
        print(answer)

if __name__ == "__main__":
    qa = WebPageQA()
    qa.run()