import re
import json
import logging
from pathlib import Path
from ...config import llama_llm

class QueryClassifier:
    """
    Uses an LLM to classify the user query as 'in-scope', 'general', or 'out-of-scope'
    based on a system prompt.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "query_classification.txt"

    def run(self, state: dict) -> dict:
        chat_history = state.get("keys", {}).get("chat_history", "").strip()
        user_response = state.get("keys", {}).get("question", "").strip()

        if not user_response:
            state.setdefault("keys", {})["classification"] = "out-of-scope"
            return state

        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading query classification prompt: {e}")
            system_prompt = ""

        classification_prompt = system_prompt.format(chat_history=chat_history, user_response=user_response)

        try:
            response = llama_llm.invoke(([("system", classification_prompt)]))
            response_text = response.content
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
        state.setdefault("keys", {})["classification"] = classification
        logging.info(f"Query classified as: {classification}")
        return state
