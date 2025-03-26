import re
import json
import logging
from pathlib import Path
from ...config import llama_llm

class QueryClassifier:
    """
    Uses an LLM to classify each sub-query as 'in-scope', 'general', or 'out-of-scope'
    based on a system prompt.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "query_classification.txt"

    def run(self, state: dict) -> dict:
        """
        Loops over each identified sub-query, classifies it, and stores the result.
        Expects sub_query_mapping["sub_queries"] to exist from sub_query_identification.
        """
        chat_history = state.get("keys", {}).get("chat_history", "").strip()
        
        # Retrieve the sub-queries from the state
        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        sub_queries = sub_query_mapping.get("sub_queries", [])

        if not sub_queries:
            logging.info("No sub-queries found. Classifying the entire user query instead.")
            # Fallback: classify the original user query if no sub-queries exist
            user_response = state.get("keys", {}).get("question", "").strip()
            classification = self.classify_query(user_response, chat_history)
            # We can store the classification at top-level or store a single item in sub_query_answers
            state["keys"]["classification"] = classification
            logging.info(f"No sub-queries found; entire query classified as: {classification}")
            return state

        # If we do have sub-queries, classify each one
        classified_sub_queries = []
        for sq in sub_queries:
            # Each sq is a string representing a "completed_query"
            classification = self.classify_query(sq, chat_history)
            classified_sub_queries.append({
                "completed_query": sq,
                "classification": classification
            })

        # Store them in the state, e.g. sub_query_mapping["classified_sub_queries"]
        sub_query_mapping["classified_sub_queries"] = classified_sub_queries
        logging.info(f"Classified sub-queries: {classified_sub_queries}")

        return state

    def classify_query(self, query_text: str, chat_history: str) -> str:
        """
        Classifies a single query (sub-query or otherwise) as 'in-scope', 'general', or 'out-of-scope'.
        """
        if not query_text.strip():
            return "out-of-scope"

        # Load the system prompt
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading query classification prompt: {e}")
            system_prompt = ""

        # Format the classification prompt
        classification_prompt = system_prompt.format(chat_history=chat_history, user_response=query_text)

        # Invoke LLM
        try:
            response = llama_llm.invoke(([("system", classification_prompt)]))
            response_text = response.content
            # Parse JSON from LLM output
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    result = json.loads(json_str)
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"JSON decoding failed: {e}")
                    result = {"classification": "out-of-scope"}
            else:
                logging.error("No JSON block found in LLM response.")
                result = {"classification": "out-of-scope"}
        except Exception as e:
            logging.error(f"Error in query classification: {e}")
            result = {"classification": "out-of-scope"}

        classification = result.get("classification", "out-of-scope").strip().lower()
        if classification not in ["in-scope", "general", "out-of-scope"]:
            classification = "out-of-scope"
        return classification
