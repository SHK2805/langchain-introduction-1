from langchain.chains.retrieval import create_retrieval_chain

from src.utils.common_utils import get_combined_chain


class QueryManager:
    def __init__(self, retriever):
        self.retriever = retriever

    def query(self, question):
        # Get the combined chain
        combine_docs_chain = get_combined_chain()
        retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        # Ask the question
        response = retrieval_chain.invoke({"input": question})
        return response['answer']