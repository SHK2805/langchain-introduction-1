from src.Config.set_config import Config
from src.utils.project_environment.envs import get_openai_llm_model_name

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")


from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def example1():
    # Source:  https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "What are everyone's favorite colors:\n\n{context}")
        ]
    )

    llm = ChatOpenAI(model=get_openai_llm_model_name())
    chain = create_stuff_documents_chain(llm, prompt)

    documents = [
        Document(page_content="Alice's favorite color is red."),
        Document(page_content="Bob's favorite color is blue."),
        Document(page_content="Charlie's favorite color is green."),
        Document(page_content="Jesse loves red but not yellow."),
        Document(page_content="Jamal loves green but not as much as he loves orange"),
        Document(page_content="Sarah loves pink but not black."),
    ]

    response = chain.invoke({"context": documents})
    print(response)

def run():
    example1()


if __name__ == "__main__":
    run()