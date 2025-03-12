"""
Module: general_out_of_scope.py
Contains two classes for handling general and out-of-scope queries
in the Langgraph pipeline. Both classes conform to a run(state) -> state interface.
"""

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
        question = state["keys"].get("question", "").strip()
        chat_history = state["keys"].get("chat_history", "").strip()

        logging.info(f"Handling general query: {question}")

        try:
            with open(self.prompt_path, "r") as f:
                general_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading general response prompt file: {e}")
            general_prompt = ""

        prompt = general_prompt.format(user_query=question)

        try:
            llm_response = llama_llm.invoke(([("system", prompt)]))
            llm_response_text = llm_response.content
        except Exception as e:
            logging.error(f"Error generating general query response: {e}")
            llm_response_text = "I'm sorry, I encountered an issue generating a response."
        
        # Ensure sub_query_mapping exists
        keys = state.setdefault("keys", {})
        sub_query_mapping = keys.setdefault("sub_query_mapping", {
            "original_query": question,
            "sub_queries": [],
            "sub_query_answers": []
        })
        
        # Append the response to sub_query_answers
        sub_query_mapping["sub_query_answers"].append({
            "completed_query": question,
            "response": llm_response_text
        })

        return state


class OutOfScopeQueryNode:
    """
    Immediately returns a fallback response for out-of-scope queries using a system prompt.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "out-of-scope.txt"

    def run(self, state: Dict) -> Dict:
        question = state["keys"].get("question", "").strip()
        chat_history = state["keys"].get("chat_history", "").strip()

        logging.warning(f"Out-of-scope query detected: {question}")

        try:
            with open(self.prompt_path, "r") as f:
                out_of_scope_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading out-of-scope prompt file: {e}")
            out_of_scope_prompt = ""

        prompt = out_of_scope_prompt.format(user_query=question)

        try:
            llm_response = llama_llm.invoke(([("system", prompt)]))
            llm_response_text = llm_response.content
        except Exception as e:
            logging.error(f"Error generating out-of-scope response: {e}")
            llm_response_text = "I'm sorry, but I can't provide information on that topic."
        
        # Ensure sub_query_mapping exists
        keys = state.setdefault("keys", {})
        sub_query_mapping = keys.setdefault("sub_query_mapping", {
            "original_query": question,
            "sub_queries": [],
            "sub_query_answers": []
        })

        sub_query_mapping["sub_query_answers"].append({
            "completed_query": question,
            "response": llm_response_text
        })

        return state
