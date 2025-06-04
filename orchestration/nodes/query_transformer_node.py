import logging
from model import TextGenerator
from langchain_core.prompts import PromptTemplate
import json


class QueryTransformer:
    """
    Takes each in-scope sub-query and, using the full context (original question,
    clarifying question, and user feedback), asks the LLM to rewrite it into
    a clearer, more answerable form.  Saves the rewritten list for downstream use.
    """

    def __init__(self):
        self.single_prompt = PromptTemplate(
            template=(
                "Rewrite this sub-query which is in arabic to be clearer and more specific so that our "
                "retrieval pipeline can find an answer:\n"
                "Sub-query: {subquery}\n\n"
                "Context:\n"
                "- Original question: {original}\n"
                "- Clarifying question: {clarifier}\n"
                "- User feedback: {feedback}\n\n"
                "Return only the rewritten sub-query in arabic, no extra text."
            ),
            input_variables=["subquery", "original", "clarifier", "feedback"],
        )
        self.generator = TextGenerator()

    def run(self, state: dict) -> dict:
        logging.info("Running QueryTransformer to rewrite unanswered sub-queries.")
        keys = state.setdefault("keys", {})
        submap = keys.get("sub_query_mapping", {})

        original = submap.get("original_query", "").strip()
        clarifier = keys.get("clarifying_question", "").strip() or "(none)"
        feedback = keys.get("user_feedback", "").strip() or "(none)"

        # Prepare the list of in-scope sub-queries
        in_scope = [
            sq["completed_query"].strip()
            for sq in submap.get("classified_sub_queries", [])
            if sq.get("classification") == "in-scope"
        ]

        rewritten = []
        for q in in_scope:
            try:
                prompt = self.single_prompt.format(
                    subquery=q,
                    original=original,
                    clarifier=clarifier,
                    feedback=feedback,
                )
                new_q = self.generator.generate(prompt, task="query_transformer", max_new_tokens=64)
                logging.info(f"Rewrote sub-query '{q}' to '{new_q}'")
            except Exception as e:
                logging.error(f"Error rewriting '{q}': {e}")
                new_q = q  # fallback to original

            rewritten.append({"original": q, "rewritten": new_q})

        # Save rewritten list for retrieval
        submap["transformed_sub_queries"] = rewritten
        keys["sub_query_mapping"] = submap
        print(
            "Transform Query Node State Output:",
            json.dumps(state, indent=2, ensure_ascii=False, default=str),
        )
        return state
