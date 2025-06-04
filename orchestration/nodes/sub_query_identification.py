import json
import logging
from pathlib import Path
from model import TextGenerator
from chat_history_manager import ChatHistoryManager
import uuid
import re

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
        self.chat_history_manager = ChatHistoryManager()
        self.generator = TextGenerator()

    async def run(self, state: dict) -> dict:
        user_query = state.get("keys", {}).get("question", "").strip()

        # 1. Check if the session_id exists, if not generate a new one
        session_id = state.get("keys", {}).get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())  # Generate a new UUID session_id
            state.setdefault("keys", {})["session_id"] = session_id
            logging.info(f"Generated new session ID: {session_id}")

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

        # 2. Load and format chat history from the ChatHistoryManager
        chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id, max_messages=5)
        prompt = sub_query_prompt.format(chat_history=chat_history_text, user_response=user_query)

        # Invoke the LLM and parse the JSON output
        try:
            response_text = self.generator.generate(prompt=prompt, task="sub_query_identification", max_new_tokens=2048)
            logging.info(response_text)
        
            response_text = self.extract_json_from_string(response_text)
            result = json.loads(response_text)
        except Exception as e:
            logging.error(f"Error in LLM invocation or JSON parsing: {e}")
            result = {}

        output = result.get("sub_queries", [])
        sub_queries = [sq.get("completed_query", "").strip() for sq in output if "completed_query" in sq]
        logging.info(sub_queries)

        # Merge with existing sub_query_mapping to preserve sub_query_answers
        sub_query_mapping = state.setdefault("keys", {}).get("sub_query_mapping", {})
        existing_answers = sub_query_mapping.get("sub_query_answers", [])

        sub_query_mapping["original_query"] = user_query
        sub_query_mapping["sub_queries"] = sub_queries
        sub_query_mapping["sub_query_answers"] = existing_answers

        state["keys"]["sub_query_mapping"] = sub_query_mapping
        logging.info(f"Sub-query mapping: {json.dumps(state['keys']['sub_query_mapping'], indent=2)}")
        return state

    def extract_json_from_string(self, text: str) -> str:
        """
        Extract the first valid JSON object from a string, ignoring surrounding text and fixing common issues.
        Returns the JSON string or an empty string if no valid JSON is found.
        """
        text = text.strip()  # Remove leading/trailing whitespace

        # Use brace counting to find a complete JSON object
        current_depth = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if current_depth == 0:
                    start = i
                current_depth += 1
            elif char == '}':
                current_depth -= 1
                if current_depth == 0 and start != -1:
                    json_candidate = text[start:i + 1]
                    # Try parsing the JSON candidate
                    try:
                        json.loads(json_candidate)
                        return json_candidate
                    except json.JSONDecodeError:
                        # If parsing fails, attempt to fix common issues
                        fixed_json = self._fix_common_json_issues(json_candidate)
                        try:
                            json.loads(fixed_json)
                            return fixed_json
                        except json.JSONDecodeError:
                            continue  # Move to next potential JSON if fixing fails

        # If no valid JSON is found, log a warning and return empty string
        logging.warning("No valid JSON found in text")
        return ""

    def _fix_common_json_issues(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues, such as missing commas between top-level keys.
        """
        # Fix missing comma between "user_query" and "sub_queries"
        pattern = r'("user_query"\s*:\s*"[^"]*")\s*("sub_queries")'
        fixed_json = re.sub(pattern, r'\1, \2', json_str)
        return fixed_json