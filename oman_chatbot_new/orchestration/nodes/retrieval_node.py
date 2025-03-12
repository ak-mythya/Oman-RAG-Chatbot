import logging
from ...config import llama_llm
from ...retrieval.retriever_setup import load_ensemble_retriever


class Retriever:
    """
    Retrieves documents using the global EnsembleRetriever.
    """

    def __init__(self):
        self.ensemble_retriever = load_ensemble_retriever()

    def run(self, state: dict) -> dict:
        logging.info("Running retrieval node.")
        question = state.get("keys", {}).get("question", "")
        if not question:
            logging.error("No question provided for retrieval.")
            state.setdefault("keys", {})["documents"] = []
            return state

        config = {"configurable": {"search_kwargs_faiss": {"k": 8}}}
        try:
            retrieved_docs = self.ensemble_retriever.invoke(question, config=config)
            docs = list(retrieved_docs)
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            docs = []

        state.setdefault("keys", {})["documents"] = docs
        return state
