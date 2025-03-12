from src.utils.project_environment.envs import *

class Config:
    def __init__(self):
        self.openai_api_key = get_open_ai_key()
        self.langchain_api_key = get_langchain_key()
        self.langsmith_v2_tracing = get_langsmith_v2_tracing()
        self.project_name = get_project_name()

    def set(self):
        try:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
            os.environ['LANGCHAIN_API_KEY'] = self.langchain_api_key
            os.environ['LANGCHAIN_PROJECT'] = self.project_name
            os.environ['LANGSMITH_TRACING_V2'] = self.langsmith_v2_tracing
        except Exception as e:
            print(f'Error setting environment variables: {e}')
            return False
        return True