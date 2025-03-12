from src.Config.set_config import Config
from langchain_openai import ChatOpenAI

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

def chat1():
    llm = ChatOpenAI(model="o3-mini")
    result = llm.invoke("What is Agentic AI?")
    return result


# example usage
if __name__ == "__main__":
    chat1_result = chat1()
    # display result
    print(chat1_result)
    # display result content / base message
    print(chat1_result.content)

