import re
import json
import logging
from pathlib import Path
from model import TextGenerator
import asyncio
import time
from chat_history_manager import ChatHistoryManager

class QueryClassifier:
    """
    Uses an LLM to classify each sub-query as 'in-scope', 'general', or 'out-of-scope',
    now with batch processing for faster classification and logging.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = (
            current_dir.parent.parent /
            "system_prompts" /
            "query_classification.txt"
        )
        self.generator = TextGenerator()
        self.chat_history_manager = ChatHistoryManager()

    async def run(self, state: dict) -> dict:
        """
        Classifies queries in batch by offloading the blocking classify_queries_batch call to a thread,
        with logging for the batch process.
        """
        session_id = state.get("keys", {}).get("session_id", "")
        if not session_id:
            logging.error("No session ID found.")
            return state

        # Retrieve and format chat history using ChatHistoryManager
        chat_history = await self.chat_history_manager.format_recent_history_as_text(session_id, max_messages=5)

        sqm = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        sub_queries = sqm.get("sub_queries", [])

        if not sub_queries:
            logging.info("No sub-queries found; classifying entire query.")
            question = state["keys"].get("question", "").strip()
            queries = [question]
            logging.info(f"queries={queries}")
        else:
            queries = sub_queries
            logging.info(f"queries={queries}")

        logging.info(f"[Classifier] Starting batch classification for {len(queries)} queries at {time.time():.3f}")
        start_time = time.time()
        classifications = await asyncio.to_thread(self.classify_queries_batch, queries, chat_history)
        end_time = time.time()
        logging.info(f"[Classifier] Batch classification completed at {end_time:.3f} (took {end_time - start_time:.3f}s)")

        if not sub_queries:
            state["keys"]["classification"] = classifications[0]
            logging.info(f"Entire query classified as: {classifications[0]}")
        else:
            sqm["classified_sub_queries"] = [
                {"completed_query": sq, "classification": cls}
                for sq, cls in zip(sub_queries, classifications)
            ]
            for sq, cls in zip(sub_queries, classifications):
                logging.info(f"[Classifier] Classified '{sq}' as: {cls}")

        return state

    def classify_queries_batch(self, queries: list[str], chat_history: str) -> list[str]:
        """
        Classifies a batch of queries using the LLM in a single batch call.
        """
        if not queries:
            return []

        # Load prompt once for all queries
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading prompt: {e}")
            system_prompt = ""

        # Prepare prompts for all queries
        prompts = [
            system_prompt.format(chat_history=chat_history, user_response=query.strip() or "out-of-scope")
            for query in queries
        ]

        # Generate classifications in batch
        try:
            generations = self.generator.generate_batch(
                prompts,
                task="query_classification",
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )
        except Exception as e:
            logging.error(f"LLM batch generation error: {e}")
            return ["out-of-scope"] * len(queries)

        # Extract classifications from generations
        classifications = []
        for gen in generations:
            try:
                json_match = re.search(r'\{.*\}', gen, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logging.error("No JSON in LLM batch response.")
                    result = {"classification": "out-of-scope"}
            except Exception as e:
                logging.error(f"Error parsing batch response: {e}")
                result = {"classification": "out-of-scope"}

            cls = result.get("classification", "out-of-scope").strip().lower()
            if cls not in ["in-scope", "general", "out-of-scope", "child-abuse","human-agent", "service-application", "exit-chat"]:
                cls = "out-of-scope"
            classifications.append(cls)

        return classifications