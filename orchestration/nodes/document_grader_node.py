

import json
import re
import logging
import asyncio
import time
from pathlib import Path
from model import TextGenerator  # Adjust import based on your setup
from langchain_core.prompts import PromptTemplate
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def extract_score(response: str) -> str:
    resp = response.strip("() \t\n\"'")
    try:
        arr = json.loads(resp)
        if isinstance(arr, list) and arr:
            cand = arr[0]
            if isinstance(cand, str):
                resp = cand.strip("() \t\n\"'")
    except Exception:
        pass

    m = re.search(r'\{\s*"score"\s*:\s*"(yes|no)"\s*\}', resp)
    if m:
        try:
            return json.loads(m.group())["score"]
        except Exception:
            pass

    if re.search(r'\byes\b', resp, re.IGNORECASE):
        return "yes"
    if re.search(r'\bno\b', resp, re.IGNORECASE):
        return "no"

    logging.warning(f"Could not extract yes/no from response: {response!r}")
    return "no"

class DocumentGrader:
    """
    Uses an LLM to grade all documents for each sub-query collectively.
    If the documents cannot answer the sub-query, a web search may be triggered based on a threshold.
    Supports batch grading for multiple sub-queries concurrently.
    """

    def __init__(self, threshold: int = 1):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "document_grader.txt"
        self.threshold = threshold
        
        self.generator = TextGenerator()

    def grade_documents(self, documents: List[Dict], query: str) -> Tuple[List[Dict], int, str]:
        """Grade all documents for a single query collectively."""
        logging.info(f"Started grade_documents for query '{query}'")

        if not documents:
            logging.info(f"No documents provided for query '{query}'")
            return [], 0

        # Build prompt input
        docs_text = "\n---\n".join(d["page_content"] for d in documents)
        tmpl = PromptTemplate(
            template=Path(self.prompt_path).read_text(encoding="utf-8"),
            input_variables=["question", "documents"],
        )
        prompt = tmpl.format(question=query, documents=docs_text)

        # # Single LLM call for all docs
        responses = self.generator.generate_batch([prompt], task="document_grading", max_new_tokens=128)
        response = responses[0].strip()
        logging.info(f"LLM raw response: {response}")
        logging.info(f"Raw LLM response for query '{query}': {response}")

        # ─── replace all of the custom parsing with a single helper call ───
        score = extract_score(response)
        # ────────────────────────────────────────────────────────────────

        # Decide which docs to keep
        if score == "yes":
            filtered = documents
            relevant_count = len(documents)
        else:
            filtered = []
            relevant_count = 0

        logging.info(
            f"Finished grade_documents for query '{query}' (kept {relevant_count}/{len(documents)})"
        )
        logging.info(f"LLM parsed score: {score}")

        return filtered, relevant_count, score
        

    async def batch_grade(
        self,
        sub_queries: List[Dict],
        queries: List[str],
        documents: List[List[Dict]]
    ) -> List[Tuple[List[Dict], int, str]]:
        """Grade documents for multiple sub-queries concurrently."""
        logging.info(f"Started batch_grade for {len(queries)} sub-queries")
        start_time = time.time()
        tasks = [
            asyncio.to_thread(self.grade_documents, docs, query)
            for query, docs in zip(queries, documents)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        graded_results: List[Tuple[List[Dict], int, str]] = []
        for sq, query, result in zip(sub_queries, queries, results):
            if isinstance(result, Exception):
                logging.error(
                    f"Failed to grade documents for query '{query}': {type(result).__name__}: {result}"
                )
                graded_results.append(([], 0, "no"))
            else:
                # result is now (filtered_docs, count, score)
                graded_results.append(result)
        logging.info(f"Graded Results Output:{graded_results}")
        end_time = time.time()
        logging.info(f"Finished batch_grade for {len(queries)} sub-queries at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
        return graded_results

    async def run(self, state: Dict) -> Dict:
        """
        Runs document grading for all sub-queries and updates the state.
        """
        logging.info("Running DocumentGrader.run")
        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        classified = sub_query_mapping.get("classified_sub_queries", [])

        queries, docs, in_scope = [], [], []
        for sq in classified:
            if sq.get("classification") == "in-scope":
                queries.append(sq.get("completed_query", ""))
                docs.append(sq.get("documents", []))
                in_scope.append(sq)

        all_filtered: List[Dict] = []
        if queries and docs:
            results = await self.batch_grade(in_scope, queries, docs)
            for sq, (filtered, count) in zip(in_scope, results):
                sq["documents"] = filtered
                sq["relevant_count"] = count
                # new flag: answerable if at least threshold docs
                sq["answerable"] = count >= self.threshold
                all_filtered.extend(filtered)
        # split into answerable vs unanswerable
        answerable = [sq for sq in classified if sq.get("answerable")]
        unanswerable = [sq for sq in classified if not sq.get("answerable")]

        # finally, write everything back into the state
        state.setdefault("keys", {})["documents"] = all_filtered
        state["keys"]["answerable_sub_queries"]   = answerable
        state["keys"]["unanswerable_sub_queries"] = unanswerable
        #state["keys"]["run_web_search"]           = "Yes" if run_search else "No"

        logging.info(
            "Document Grader Node State Output:\n%s",
            json.dumps(state, indent=2, ensure_ascii=False)
        )
        return state
        #return state


if __name__ == "__main__":
    import asyncio

    # 1) Create a dummy sub-query and some fake docs
    docs = [
        {"page_content": "هذا نص عن كيفية دمج الجمعيات...", "metadata": {"score": 600}},
        {"page_content": "شرح لإجراءات الدمج حسب القانون...", "metadata": {"score": 620}},
    ]
    query = "وفقاً للقانون كيف تتم عملية دمج جمعيات المجتمع المدني؟"

    # 2) Run the synchronous grader directly
    grader = DocumentGrader(threshold=1)
    filtered, count = grader.grade_documents(docs, query)
    print(f"\nStandalone grade_documents → kept {count} docs:")
    for d in filtered:
        print(" •", d["page_content"][:50])

    # 3) Or test the async batch_grade
    async def test_batch():
        sub_queries = [{"completed_query": query, "classification": "in-scope"}]
        results = await grader.batch_grade(sub_queries, [query], [docs])
        print("\nStandalone batch_grade results:", results)

    asyncio.run(test_batch())