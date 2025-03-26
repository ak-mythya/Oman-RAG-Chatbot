import json
import re
import logging
from ...config import llama_llm
from pathlib import Path
from langchain_core.prompts import PromptTemplate

class DocumentGrader:
    """
    Uses an LLM to grade each document for relevance for each sub-query.
    If fewer than a threshold number of documents are graded "yes", a web search will be triggered.
    """

    def __init__(self, threshold: int = 3):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "document_grader.txt"
        self.threshold = threshold

    def run(self, state: dict) -> dict:
        logging.info("Running document grading node.")
        
        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
        
        # Process each sub-query's documents
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")
            documents = sq_data.get("documents", [])
            
            if classification == "in-scope":
                logging.info(f"Grading documents for sub-query: {sq_text}")

                # Grade the documents for the in-scope sub-query
                filtered_docs, relevant_count = self.grade_documents(documents, sq_text)

                # Update the documents for the sub-query
                sq_data["documents"] = filtered_docs
                sq_data["relevant_count"] = relevant_count

        # Check if we need to run a web search based on the number of relevant documents
        run_web_search = "Yes" if any(sq.get("relevant_count", 0) < self.threshold for sq in classified_sub_queries) else "No"
        state["keys"]["run_web_search"] = run_web_search

        return state

    def grade_documents(self, documents, query):
        """
        Helper method to grade documents using the provided query.
        """
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
            prompt = prompt_tmpl.format(context=d["page_content"], question=query)
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

        return filtered_docs, relevant_count
