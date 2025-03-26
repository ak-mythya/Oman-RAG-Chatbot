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
        # Fetch the sub-query data from the state
        sub_query_mapping = state.setdefault("keys", {}).get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        # Only transform in-scope sub-queries
        for sq_data in classified_sub_queries:
            classification = sq_data.get("classification", "out-of-scope")
            
            if classification == "in-scope":
                original_query = sq_data.get("completed_query", "")
                needs_transformation = sq_data.get("needs_transformation", False)  # Check if transformation is needed

                if needs_transformation:
                    # Run transformation
                    prompt = self.prompt_template.format(question=original_query)
                    try:
                        transformed_query = llama_llm.invoke(prompt, max_tokens=256, temperature=0).content
                        sq_data["transformed_query"] = transformed_query
                        logging.info(f"Transformed query: {transformed_query}")
                    except Exception as e:
                        logging.error(f"Error transforming query: {e}")
                        sq_data["transformed_query"] = original_query  # Fallback to original query
                else:
                    sq_data["transformed_query"] = original_query  # If no transformation, use original query
            else:
                sq_data["transformed_query"] = ""  # If out-of-scope or general, no transformation

        # Update state
        state["keys"]["sub_query_mapping"] = sub_query_mapping
        return state
