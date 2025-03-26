import logging
import json
from typing import Dict
from pathlib import Path
from ...config import llama_llm  # Use your LLM for response generation

class GeneralQueryNode:
    """
    Handles general queries by generating responses without retrieval.
    Uses a system prompt (general_responses.txt) to guide the response.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "general_responses.txt"
        
    def run(self, state: Dict) -> Dict:
        logging.info("Handling general queries.")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        # For each classified sub-query, generate a response
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")
            
            if classification == "general":
                logging.info(f"Handling general sub-query: {sq_text}")
                
                try:
                    with open(self.prompt_path, "r") as f:
                        general_prompt = f.read()
                except Exception as e:
                    logging.error(f"Error reading general response prompt file: {e}")
                    general_prompt = ""
                
                prompt = general_prompt.format(user_query=sq_text)

                try:
                    llm_response = llama_llm.invoke(([("system", prompt)]))
                    llm_response_text = llm_response.content
                except Exception as e:
                    logging.error(f"Error generating general sub-query response: {e}")
                    llm_response_text = "I'm sorry, I encountered an issue generating a response."

                # Store response in the sub-query answers
                sq_data["response"] = llm_response_text

        # Update the state with the responses for each general sub-query
        state["keys"]["sub_query_mapping"]["sub_query_answers"] = [
            {"completed_query": sq.get("completed_query", ""), "response": sq.get("response", "")}
            for sq in classified_sub_queries if sq.get("response")
        ]
        
        return state


class OutOfScopeQueryNode:
    """
    Immediately returns a fallback response for out-of-scope queries using a system prompt.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "out-of-scope.txt"

    def run(self, state: Dict) -> Dict:
        logging.warning("Handling out-of-scope queries.")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        # For each classified sub-query, generate a response
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")
            
            if classification == "out-of-scope":
                logging.info(f"Handling out-of-scope sub-query: {sq_text}")
                
                try:
                    with open(self.prompt_path, "r") as f:
                        out_of_scope_prompt = f.read()
                except Exception as e:
                    logging.error(f"Error reading out-of-scope response prompt file: {e}")
                    out_of_scope_prompt = ""

                prompt = out_of_scope_prompt.format(user_query=sq_text)

                try:
                    llm_response = llama_llm.invoke(([("system", prompt)]))
                    llm_response_text = llm_response.content
                except Exception as e:
                    logging.error(f"Error generating out-of-scope sub-query response: {e}")
                    llm_response_text = "I'm sorry, but I can't provide information on that topic."

                # Store response in the sub-query answers
                sq_data["response"] = llm_response_text

        # Update the state with the responses for each out-of-scope sub-query
        state["keys"]["sub_query_mapping"]["sub_query_answers"] = [
            {"completed_query": sq.get("completed_query", ""), "response": sq.get("response", "")}
            for sq in classified_sub_queries if sq.get("response")
        ]

        return state
