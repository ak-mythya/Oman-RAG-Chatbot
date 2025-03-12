import logging
from ...config import llama_llm
from langchain_core.prompts import PromptTemplate

class QueryTransformer:
    """
    Transforms the user's question into a search-optimized version.
    """

    def __init__(self):
        self.prompt_template = PromptTemplate(
            template="""Generate a search-optimized version of this question by analyzing its core semantic meaning and intent.
Return only the improved question with no additional text:
-------
{question}
-------
""",
            input_variables=["question"]
        )

    def run(self, state: dict) -> dict:
        logging.info("Running query transformer node.")
        question = state.get("keys", {}).get("question", "")
        prompt = self.prompt_template.format(question=question)
        try:
            better_question = llama_llm.invoke(prompt, max_tokens=256, temperature=0).content
            logging.info(f"Improved question: {better_question}")
        except Exception as e:
            logging.error(f"Error transforming query: {e}")
            better_question = question
        state.setdefault("keys", {})["question"] = better_question
        state["keys"]["did_transform"] = True
        return state
