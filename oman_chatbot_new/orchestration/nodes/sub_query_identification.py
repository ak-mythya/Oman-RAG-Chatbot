import json
import logging
from pathlib import Path
from ...config import llama_llm

class SubQueryIdentifier:
    """
    Uses the LLM to identify and complete sub-queries from the original user query.

    Expects LLM output in JSON format:
      {
        "user_query": "<user_query>",
        "sub_queries": [
          {"completed_query": "<refined query>", "justification": "<explanation>"},
          ...
        ]
      }
    The node extracts the "completed_query" values and stores them in the state.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "sub-queries.txt"

    def run(self, state: dict) -> dict:
        user_query = state.get("keys", {}).get("question", "").strip()
        if not user_query:
            logging.error("Empty user query in sub-query identification.")
            # Merge logic: preserve any existing sub_query_answers
            sub_query_mapping = state.setdefault("keys", {}).get("sub_query_mapping", {})
            existing_answers = sub_query_mapping.get("sub_query_answers", [])

            sub_query_mapping["original_query"] = user_query
            sub_query_mapping["sub_queries"] = []
            sub_query_mapping["sub_query_answers"] = existing_answers

            state["keys"]["sub_query_mapping"] = sub_query_mapping
            return state

        # Read the sub-query prompt from file
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                sub_query_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading sub-query prompt file: {e}")
            sub_query_prompt = ""

        chat_history = state.get("keys", {}).get("chat_history", "")
        prompt = sub_query_prompt.format(chat_history=chat_history, user_response=user_query)

        # Invoke the LLM and parse the JSON output
        try:
            response = llama_llm.invoke(([("system", prompt)]))
            response_text = response.content
            result = json.loads(response_text)
        except Exception as e:
            logging.error(f"Error in LLM invocation or JSON parsing: {e}")
            result = {}

        output = result.get("sub_queries", [])
        sub_queries = [sq.get("completed_query", "").strip() for sq in output if "completed_query" in sq]

        # Merge with existing sub_query_mapping to preserve sub_query_answers
        sub_query_mapping = state.setdefault("keys", {}).get("sub_query_mapping", {})
        existing_answers = sub_query_mapping.get("sub_query_answers", [])

        sub_query_mapping["original_query"] = user_query
        sub_query_mapping["sub_queries"] = sub_queries
        sub_query_mapping["sub_query_answers"] = existing_answers

        state["keys"]["sub_query_mapping"] = sub_query_mapping
        logging.info(f"Sub-query mapping: {json.dumps(state['keys']['sub_query_mapping'], indent=2)}")
        return state
