import os
from dotenv import load_dotenv
load_dotenv()

def get_key(key_name):
    return os.getenv(key_name)

def get_open_ai_key():
    return get_key('OPENAI_API_KEY')

def get_langchain_key():
    return get_key('LANGCHAIN_API_KEY')

def get_project_name():
    return get_key('LANGCHAIN_PROJECT')

def get_langsmith_v2_tracing():
    return get_key('LANGSMITH_TRACING_V2')

def get_openai_llm_model_name():
    return get_key('OPENAI_LLM_MODEL_NAME')

