import json
import re
import logging
from ...config import llama_llm
from pathlib import Path
from langchain_core.prompts import PromptTemplate

class DocumentGrader:
    """
    Uses an LLM to grade each document for relevance.
    If fewer than a threshold number of documents are graded "yes", a web search will be triggered.
    """

    def __init__(self, threshold: int = 3):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "document_grader.txt"
        self.threshold = threshold

    def run(self, state: dict) -> dict:
        logging.info("Running document grading node.")
        question = state.get("keys", {}).get("question", "")
        documents = state.get("keys", {}).get("documents", [])
        try:
            with open(self.prompt_path, "r") as f:
                grading_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading document grader prompt: {e}")
            grading_prompt = ""

        prompt_tmpl = PromptTemplate(
            template=grading_prompt,
            input_variables=["context", "question"]
        )

        filtered_docs = []
        relevant_count = 0

        for d in documents:
            prompt = prompt_tmpl.format(context=d.page_content, question=question)
            try:
                response = llama_llm.invoke(prompt, max_tokens=256, temperature=0)
                response_content = response.content
                json_match = re.search(r'\{.*\}', response_content)
                if json_match:
                    response_json_str = json_match.group()
                    score = json.loads(response_json_str)
                    if score.get("score") == "yes":
                        filtered_docs.append(d)
                        relevant_count += 1
                else:
                    filtered_docs.append(d)
                    relevant_count += 1
            except Exception as e:
                logging.error(f"Error grading document: {e}")
                filtered_docs.append(d)
                relevant_count += 1

        run_web_search = "Yes" if relevant_count < self.threshold else "No"
        state.setdefault("keys", {})["documents"] = filtered_docs
        state["keys"]["run_web_search"] = run_web_search
        return state
