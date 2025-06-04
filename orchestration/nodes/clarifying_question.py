import logging
from model import TextGenerator
from langchain_core.prompts import PromptTemplate
import json
from langgraph.types import interrupt


class Clarifying_Question:
    """
    Takes the original_query + ALL sub-queries (tagged with their classification),
    tells the LLM: “the RAG pipeline couldn’t answer the in-scope parts, so focus only
    on those and the original question to generate one clarifying question,”
    and writes it into state["keys"]["clarifying_question"].
    """

    def __init__(self):
        self.prompt_template = PromptTemplate(
            template=(
                # 0) Persona: you’re a clarifying-question assistant
                "You are a clarifying-question assistant whose sole task is to pose one concise clarifying question.\n\n"
                # 1) Explain why we’re here
                "Our retrieval pipeline tried to answer the user’s question but failed to "
                "resolve the parts we care about.  We now need one more question to "
                "pin down exactly what the user means.\n\n"
                # 2) Give them the original for context
                "ORIGINAL QUESTION:\n"
                "{original}\n\n"
                # 3) Show all sub-queries, tagged
                "ALL SUB-QUERIES (with labels):\n"
                "{all_tagged}\n\n"
                # 4) Very explicit focus instruction
                "📌 **IMPORTANT**: You must ignore every sub-query labeled “out-of-scope” "
                "or “general.”  Use **only** the ORIGINAL QUESTION and the [in-scope] items.\n\n"
                # 5) Tell them format and tone
                "Ask clarification question in arabic"
                "Write exactly one **short**, **clear**, **follow-up question** that will "
                "help us capture the user’s missing detail.  Return **only** that question—"
                "no bullets, no numbering, no explanation.\n\n"
                # 6) The anchor where they fill in the result
                "Clarifying Question:"
            ),
            input_variables=["original", "all_tagged"],
        )
        self.generator = TextGenerator()

    def run(self, state: dict) -> dict:
        logging.info("Running Clarifying_Question node.")

        keys = state.setdefault("keys", {})
        sub_map = keys.get("sub_query_mapping", {})

        original = sub_map.get("original_query", "").strip()

        # Build a tagged list of ALL sub-queries
        classified = sub_map.get("classified_sub_queries", [])
        tagged_lines = []
        for sq in classified:
            q = sq.get("completed_query", "").strip()
            cls = sq.get("classification", "general")
            if q:
                tagged_lines.append(f"- [{cls}] {q}")
        all_tagged = "\n".join(tagged_lines) or "(no sub-queries found)"

        # Fill in the prompt
        prompt = self.prompt_template.format(
            original=original or "(none provided)", all_tagged=all_tagged
        )
        logging.info(f"Prompt for LLM: {prompt}")

        # Call the LLM
        try:
            clarifying = self.generator.generate(prompt, task="clarifying_question", max_new_tokens=128)
            logging.info(f"Clarifying question from LLM: {clarifying}")
        except Exception as e:
            logging.error(f"Error generating clarifying question: {e}")
            # fallback: use the original if only one, otherwise join in-scope with “/”
            in_scope = [
                sq["completed_query"].strip()
                for sq in classified
                if sq.get("classification") == "in-scope" and sq.get("completed_query")
            ]
            if len(in_scope) == 1:
                clarifying = in_scope[0]
            else:
                clarifying = " / ".join(in_scope) or original

        # Write it back
        # keys["clarifying_question"] = clarifying

        # 6) Write it back into state
        state.setdefault("keys", {})["clarifying_question"] = clarifying
        # print(
        #     "Clarifying Subquery Node State Output:",
        #     json.dumps(state, indent=2, ensure_ascii=False),
        # )

        return state

        # **This return pauses the graph** and hands the clarifier back to your API/UI
        # return interrupt(clarifying)
        # return interrupt( clarifying)
