import logging
import json
import random
from typing import Dict, Optional, List
from pathlib import Path
from model import TextGenerator
from sentence_transformers import SentenceTransformer, util
from chat_history_manager import ChatHistoryManager
from config import EMBEDDING_MODEL
import re

# Intent examples for similarity-based classification
INTENT_EXAMPLES = {
    "greeting": ["مرحباً", "أهلاً", "صباح الخير"],
    "goodbye": ["وداعاً", "إلى اللقاء", "أراك لاحقاً"],
    "thanks": ["شكراً", "ممتن لك", "أشكرك"]
}

# Multiple responses per intent
RESPONSES = {
    "greeting": [
        "مرحباً! كيف يمكنني مساعدتك اليوم؟",
        "أهلاً وسهلاً بك! ما الذي يمكنني فعله لأجلك؟",
        "صباح الخير! كيف أستطيع خدمتك؟"
    ],
    "goodbye": [
        "وداعاً! أتمنى لك يوماً رائعاً!",
        "إلى اللقاء! لا تتردد في العودة إذا احتجت شيئاً.",
        "مع السلامة! كن بخير."
    ],
    "thanks": [
        "على الرحب والسعة! أخبرني إذا كان لديك أي استفسار آخر.",
        "لا شكر على واجب!",
        "سعيد بأنني استطعت المساعدة."
    ]
}

# # Prepare sentence transformer and intent embeddings
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[2]  # oman_chatbot_new
# MODELS_DIR = ROOT / "models"

# def snapshot(repo: str) -> Path:
#     return MODELS_DIR / repo

# SENT_DIR = snapshot("models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")

EMBEDDING_MODEL = EMBEDDING_MODEL
INTENT_EMBEDDINGS = {
    intent: EMBEDDING_MODEL.encode(examples, convert_to_tensor=True)
    for intent, examples in INTENT_EXAMPLES.items()
}

def match_intents_batch(texts: List[str]) -> List[Optional[str]]:
    results = []
    for text in texts:
        query_embedding = EMBEDDING_MODEL.encode(text, convert_to_tensor=True)
        best_intent = None
        best_score = 0.0
        for intent, example_embeddings in INTENT_EMBEDDINGS.items():
            scores = util.cos_sim(query_embedding, example_embeddings)
            max_score = scores.max().item()
            if max_score > best_score:
                best_score = max_score
                best_intent = intent
        results.append(best_intent if best_score > 0.6 else None)
    return results

class GeneralQueryNode:
    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = (
            current_dir.parent.parent / "system_prompts" / "general_responses.txt"
        )
        self.generator = TextGenerator()
        self.chat_history_manager = ChatHistoryManager()

    async def run(self, state: Dict) -> Dict:
        logging.info("Handling general queries.")

        sub_query_mapping = state.setdefault("keys", {}).setdefault(
            "sub_query_mapping", {}
        )
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
        session_id = state.get("keys", {}).get("session_id", "")

        # Retrieve and format chat history for the session
        chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id, max_messages=5)

        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                general_prompt_template = f.read()
        except Exception as e:
            logging.error(
                f"Error reading general response prompt file: {e}. Using empty prompt."
            )
            general_prompt_template = ""

        general_queries = [
            sq_data for sq_data in classified_sub_queries
            if sq_data.get("classification", "out-of-scope") == "general"
        ]
        logging.info(f"general_queries={general_queries}")
        if not general_queries:
            logging.info("No general sub-queries to process.")
            state["keys"]["sub_query_mapping"]["sub_query_answers"] = []
            return state

        query_texts = [sq.get("completed_query", "") for sq in general_queries]
        logging.info(f"Processing {len(query_texts)} general sub-queries.")

        intent_matches = match_intents_batch(query_texts)

        llm_needed = []
        for sq_data, intent, query_text in zip(general_queries, intent_matches, query_texts):
            if intent:
                sq_data["response"] = random.choice(RESPONSES[intent])
                logging.info(f"Intent '{intent}' matched for '{query_text}' → fast response.")
            else:
                llm_needed.append(sq_data)

        def extract_json_or_return(text: str) -> str:
            # Regex to find the first JSON object in the text
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return match.group(1)
            return text

        if llm_needed:
            prompts = [
                general_prompt_template.format(chat_history=chat_history_text, user_query=sq["completed_query"])
                for sq in llm_needed
            ]
            try:
                responses = self.generator.generate_batch(
                    prompts,
                    task="general_query",
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
                for sq, response in zip(llm_needed, responses):
                    sq["response"] = extract_json_or_return(response)
            except Exception as e:
                logging.error(f"Error in batch generation: {e}")
                for sq in llm_needed:
                    sq["response"] = "Error during batch generation."

        state["keys"]["sub_query_mapping"]["sub_query_answers"] = [
            {
                "completed_query": sq.get("completed_query", ""),
                "response": sq.get("response", ""),
            }
            for sq in classified_sub_queries
            if "response" in sq
        ]
        # …after populating sub_query_mapping["sub_query_answers"]…
        keys = state["keys"]
        # Flatten all sub-query responses into one string (or pick first)
        keys["classification"] = "general"
        keys["response"] = "\n".join(
            ans["response"]
            for ans in keys["sub_query_mapping"]["sub_query_answers"]
        )

        return state