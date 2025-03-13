from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.utils.project_environment.envs import get_openai_llm_model_name

# Run the main.py to set the Config
llm = ChatOpenAI(model=get_openai_llm_model_name())
def chat1():
    result = llm.invoke("What is Agentic AI?")
    return result

def run_chat1():
    chat1_result = chat1()
    # display result
    print(chat1_result)
    # display result content / base message
    print(chat1_result.content)

def prompt_chat1(human_message):
    str_output_parser = StrOutputParser()
    template = ChatPromptTemplate(
      [
          ("system", "You are an expert in hindu Mythology. You are a world renewed professor. You name is Rambantu."),
          ("user", "{human_message}")
      ]
    )

    chain = template | llm | str_output_parser
    response = chain.invoke({"human_message": human_message})
    return response

def prompt_chat_json(query):
    json_output_parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": json_output_parser.get_format_instructions()},
    )

    chain = prompt | llm | json_output_parser
    response = chain.invoke({"query": query})
    return response



# example usage
if __name__ == "__main__":
    print(prompt_chat_json("In two sentences tell me about Arjuna's bow."))

