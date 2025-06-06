

# import json
# import re
# import logging
# import asyncio
# import time
# from pathlib import Path
# from model import TextGenerator  # Adjust import based on your setup
# from langchain_core.prompts import PromptTemplate
# from typing import List, Tuple, Dict

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )


# def extract_score(response: str) -> str:
#     resp = response.strip("() \t\n\"'")
#     try:
#         arr = json.loads(resp)
#         if isinstance(arr, list) and arr:
#             cand = arr[0]
#             if isinstance(cand, str):
#                 resp = cand.strip("() \t\n\"'")
#     except Exception:
#         pass

#     m = re.search(r'\{\s*"score"\s*:\s*"(yes|no)"\s*\}', resp)
#     if m:
#         try:
#             return json.loads(m.group())["score"]
#         except Exception:
#             pass

#     if re.search(r'\byes\b', resp, re.IGNORECASE):
#         return "yes"
#     if re.search(r'\bno\b', resp, re.IGNORECASE):
#         return "no"

#     logging.warning(f"Could not extract yes/no from response: {response!r}")
#     return "no"

# class DocumentGrader:
#     """
#     Uses an LLM to grade all documents for each sub-query collectively.
#     If the documents cannot answer the sub-query, a web search may be triggered based on a threshold.
#     Supports batch grading for multiple sub-queries concurrently.
#     """

#     def __init__(self, threshold: int = 1):
#         current_dir = Path(__file__).resolve().parent
#         self.prompt_path = current_dir.parent.parent / "system_prompts" / "document_grader.txt"
#         self.threshold = threshold
        
#         self.generator = TextGenerator()

#     def grade_documents(self, documents: List[Dict], query: str) -> Tuple[List[Dict], int, str]:
#         """Grade all documents for a single query collectively."""
#         logging.info(f"Started grade_documents for query '{query}'")

#         if not documents:
#             logging.info(f"No documents provided for query '{query}'")
#             return [], 0

#         # Build prompt input
#         docs_text = "\n---\n".join(d["page_content"] for d in documents)
#         tmpl = PromptTemplate(
#             template=Path(self.prompt_path).read_text(encoding="utf-8"),
#             input_variables=["question", "documents"],
#         )
#         prompt = tmpl.format(question=query, documents=docs_text)

#         # # Single LLM call for all docs
#         responses = self.generator.generate_batch([prompt], task="document_grading", max_new_tokens=128)
#         response = responses[0].strip()
#         logging.info(f"LLM raw response: {response}")
#         logging.info(f"Raw LLM response for query '{query}': {response}")

#         # ─── replace all of the custom parsing with a single helper call ───
#         score = extract_score(response)
#         # ────────────────────────────────────────────────────────────────

#         # Decide which docs to keep
#         if score == "yes":
#             filtered = documents
#             relevant_count = len(documents)
#         else:
#             filtered = []
#             relevant_count = 0

#         logging.info(
#             f"Finished grade_documents for query '{query}' (kept {relevant_count}/{len(documents)})"
#         )
#         logging.info(f"LLM parsed score: {score}")

#         return filtered, relevant_count, score
        

#     async def batch_grade(
#         self,
#         sub_queries: List[Dict],
#         queries: List[str],
#         documents: List[List[Dict]]
#     ) -> List[Tuple[List[Dict], int, str]]:
#         """Grade documents for multiple sub-queries concurrently."""
#         logging.info(f"Started batch_grade for {len(queries)} sub-queries")
#         start_time = time.time()
#         tasks = [
#             asyncio.to_thread(self.grade_documents, docs, query)
#             for query, docs in zip(queries, documents)
#         ]
#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         graded_results: List[Tuple[List[Dict], int, str]] = []
#         for sq, query, result in zip(sub_queries, queries, results):
#             if isinstance(result, Exception):
#                 logging.error(
#                     f"Failed to grade documents for query '{query}': {type(result).__name__}: {result}"
#                 )
#                 graded_results.append(([], 0, "no"))
#             else:
#                 # result is now (filtered_docs, count, score)
#                 graded_results.append(result)
#         logging.info(f"Graded Results Output:{graded_results}")
#         end_time = time.time()
#         logging.info(f"Finished batch_grade for {len(queries)} sub-queries at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
#         return graded_results

    # async def run(self, state: Dict) -> Dict:
    #     """
    #     Runs document grading for all sub-queries and updates the state.
    #     """
    #     logging.info("Running DocumentGrader.run")
    #     sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
    #     classified = sub_query_mapping.get("classified_sub_queries", [])

    #     queries, docs, in_scope = [], [], []
    #     for sq in classified:
    #         if sq.get("classification") == "in-scope":
    #             queries.append(sq.get("completed_query", ""))
    #             docs.append(sq.get("documents", []))
    #             in_scope.append(sq)

    #     all_filtered: List[Dict] = []
    #     if queries and docs:
    #         results = await self.batch_grade(in_scope, queries, docs)
    #         for sq, (filtered, count) in zip(in_scope, results):
    #             sq["documents"] = filtered
    #             sq["relevant_count"] = count
    #             # new flag: answerable if at least threshold docs
    #             sq["answerable"] = count >= self.threshold
    #             all_filtered.extend(filtered)
    #     # split into answerable vs unanswerable
    #     answerable = [sq for sq in classified if sq.get("answerable")]
    #     unanswerable = [sq for sq in classified if not sq.get("answerable")]

    #     # finally, write everything back into the state
    #     state.setdefault("keys", {})["documents"] = all_filtered
    #     state["keys"]["answerable_sub_queries"]   = answerable
    #     state["keys"]["unanswerable_sub_queries"] = unanswerable
    #     #state["keys"]["run_web_search"]           = "Yes" if run_search else "No"

    #     logging.info(
    #         "Document Grader Node State Output:\n%s",
    #         json.dumps(state, indent=2, ensure_ascii=False)
    #     )
    #     return state
#         #return state


# if __name__ == "__main__":
#     import asyncio

#     # 1) Create a dummy sub-query and some fake docs
#     docs = [
#         {"page_content": "هذا نص عن كيفية دمج الجمعيات...", "metadata": {"score": 600}},
#         {"page_content": "شرح لإجراءات الدمج حسب القانون...", "metadata": {"score": 620}},
#     ]
#     query = "وفقاً للقانون كيف تتم عملية دمج جمعيات المجتمع المدني؟"

#     # 2) Run the synchronous grader directly
#     grader = DocumentGrader(threshold=1)
#     filtered, count = grader.grade_documents(docs, query)
#     print(f"\nStandalone grade_documents → kept {count} docs:")
#     for d in filtered:
#         print(" •", d["page_content"][:50])

#     # 3) Or test the async batch_grade
#     async def test_batch():
#         sub_queries = [{"completed_query": query, "classification": "in-scope"}]
#         results = await grader.batch_grade(sub_queries, [query], [docs])
#         print("\nStandalone batch_grade results:", results)

#     asyncio.run(test_batch())

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
    """Robustly extract a 'yes' or 'no' score from an LLM response."""
    logging.debug(f"Raw LLM response: {response!r}")

    # Strip code blocks or markdown if present
    response = response.strip().strip("`")

    # Try JSON parsing first
    try:
        parsed = json.loads(response)
        logging.debug(f"Parsed JSON: {parsed!r}")

        if isinstance(parsed, dict) and "score" in parsed:
            return parsed["score"].strip().lower()
        if isinstance(parsed, str):
            if parsed.lower() in ("yes", "no"):
                return parsed.lower()
        if isinstance(parsed, list) and parsed:
            inner = parsed[0]
            if isinstance(inner, dict) and "score" in inner:
                return inner["score"].strip().lower()
            if isinstance(inner, str) and inner.lower() in ("yes", "no"):
                return inner.lower()
    except Exception as e:
        logging.debug(f"Failed JSON parse: {e}")

    # Regex match fallback
    match = re.search(r'"score"\s*:\s*"(yes|no)"', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Final fallback
    if re.search(r'\byes\b', response, re.IGNORECASE):
        return "yes"
    if re.search(r'\bno\b', response, re.IGNORECASE):
        return "no"

    logging.warning(f"Could not extract score from: {response!r}")
    return "no"

class DocumentGrader:
    """
    Two-step document grading:
    Step 1: Grade each document individually for relevance 
    Step 2: Grade relevant documents collectively for answer capability
    """

    def __init__(self, threshold: int = 1):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "document_grader.txt"
        self.threshold = threshold
        self.generator = TextGenerator()

    def step1_individual_grading(self, documents: List[Dict], query: str) -> Tuple[List[Dict], List[str]]:
        """Step 1: Grade each document individually for relevance"""
        logging.info(f"Step 1: Individual grading for query '{query}' - {len(documents)} documents")
        
        auto, to_grade = [], []
        all_grades = []

        # Prepare prompts for individual grading
        tmpl = PromptTemplate(
            template=Path(self.prompt_path).read_text(encoding="utf-8"),
            input_variables=["documents", "question"],
        )

        prompts = [tmpl.format(documents=d["page_content"], question=query) for d in documents]
        individual_grades = []
        
        # Batch process individual document grading
        if prompts:
            responses = self.generator.generate_batch(prompts, task="document_grading", max_new_tokens=128)
            logging.info(f"\n\n\n\n\nResponses: {responses}\n\n\n\n\n")
            for d, resp in zip(documents, responses):
                score = extract_score(resp)
                individual_grades.append(score)
                if score == "yes":
                    auto.append(d)  # Add to relevant documents
        
        
        logging.info(f"Step 1 complete: {len(auto)}/{len(documents)} documents passed individual relevance")
        return auto, individual_grades

    def step2_collective_grading(self, relevant_documents: List[Dict], query: str) -> Tuple[List[Dict], str]:
        """Step 2: Grade relevant documents collectively for answer capability"""
        logging.info(f"Step 2: Collective grading for query '{query}' - {len(relevant_documents)} relevant documents")
        
        if not relevant_documents:
            logging.info(f"No relevant documents from Step 1 for query '{query}'")
            return [], "no"

        # Build collective prompt
        docs_text = "\n---\n".join(d["page_content"] for d in relevant_documents)
        
        # Use same prompt template but adapt for collective grading
        prompt_content = Path(self.prompt_path).read_text(encoding="utf-8")
        
        # Check if prompt uses "documents" or "context" variable
        
        tmpl = PromptTemplate(
                template=prompt_content,
                input_variables=["question", "documents"],
            )
        prompt = tmpl.format(question=query, documents=docs_text)
        
        # Single LLM call for collective decision
        responses = self.generator.generate_batch([prompt], task="document_grading", max_new_tokens=128)
        response = responses[0].strip()
        
        logging.info(f"Step 2 LLM raw response: {response}")
        collective_score = extract_score(response)
        
        # Return documents only if collective score is "yes"
        if collective_score == "yes":
            final_documents = relevant_documents
        else:
            final_documents = []
            
        logging.info(f"Step 2 complete: Collective score '{collective_score}' → {len(final_documents)} final documents")
        return final_documents, collective_score

    def grade_documents_two_step(self, documents: List[Dict], query: str) -> Tuple[List[Dict], int, Dict]:
        """Combined 2-step grading: individual relevance → collective answer capability"""
        start_time = time.time()
        logging.info(f"Starting 2-step grading for query '{query}' with {len(documents)} documents")
        
        # Step 1: Individual document relevance
        step1_relevant_docs, step1_individual_grades = self.step1_individual_grading(documents, query)
        
        # Step 2: Collective answer capability
        final_documents, step2_collective_score = self.step2_collective_grading(step1_relevant_docs, query)
        
        # Prepare detailed grading results for evaluation
        grading_details = {
            "step1_individual_grades": step1_individual_grades,
            "step1_relevant_documents": step1_relevant_docs,
            "step1_relevant_count": len(step1_relevant_docs),
            "step2_collective_score": step2_collective_score,
            "step2_final_documents": final_documents,
            "step2_final_count": len(final_documents)
        }
        
        end_time = time.time()
        logging.info(f"2-step grading complete for '{query}': {len(documents)} → {len(step1_relevant_docs)} → {len(final_documents)} (Duration: {end_time - start_time:.2f}s)")
        
        return final_documents, len(final_documents), grading_details

    async def batch_grade(self, sub_queries: List[Dict], queries: List[str], documents: List[List[Dict]]) -> List[Tuple[List[Dict], int, Dict]]:
        """Grade documents for multiple sub-queries using 2-step process concurrently"""
        start_time = time.time()
        logging.info(f"Started 2-step batch_grade for {len(queries)} sub-queries")
        
        tasks = [
            asyncio.to_thread(self.grade_documents_two_step, docs, query)
            for query, docs in zip(queries, documents)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and prepare results
        graded_results = []
        for sq, query, result in zip(sub_queries, queries, results):
            if isinstance(result, Exception):
                logging.error(f"Failed 2-step grading for query '{query}': {type(result).__name__}: {result}")
                # Return empty results on failure
                graded_results.append(([], 0, {
                    "step1_individual_grades": [],
                    "step1_relevant_documents": [],
                    "step1_relevant_count": 0,
                    "step2_collective_score": "no",
                    "step2_final_documents": [],
                    "step2_final_count": 0
                }))
            else:
                graded_results.append(result)

        end_time = time.time()
        logging.info(f"Finished 2-step batch_grade for {len(queries)} sub-queries (Duration: {end_time - start_time:.2f}s)")
        return graded_results

    async def run(self, state: dict) -> dict:
        """
        Runs 2-step document grading for all sub-queries and updates the state.
        This version follows exactly the same in-place update logic as the previous working grader.
        """
        start_time = time.time()
        logging.info(f"Started 2-step DocumentGrader.run at {start_time:.2f}")

        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        classified = sub_query_mapping.get("classified_sub_queries", [])

        # 1. Collect in-scope sub-queries and their documents
        queries: List[str] = []
        docs: List[List[Dict]] = []
        in_scope: List[Dict] = []
        for sq in classified:
            if sq.get("classification") == "in-scope":
                queries.append(sq.get("completed_query", ""))
                docs.append(sq.get("documents", []))
                in_scope.append(sq)

        all_filtered: List[Dict] = []
        # 2. If there are any in-scope queries, run the two-step grader on them
        if queries and docs:
            graded_results = await self.batch_grade(in_scope, queries, docs)

            # 3. Update each sq in the original `classified` list (in-place)
            for sq, (filtered_docs, count, grading_details) in zip(in_scope, graded_results):
                # Step 1 / Step 2 metadata
                sq["step1_individual_grades"] = grading_details["step1_individual_grades"]
                sq["step1_relevant_documents"] = grading_details["step1_relevant_documents"]
                sq["step1_relevant_count"] = grading_details["step1_relevant_count"]
                sq["step2_collective_score"] = grading_details["step2_collective_score"]
                # Final results
                sq["documents"] = filtered_docs
                sq["relevant_count"] = count
                sq["answerable"] = count >= self.threshold

                all_filtered.extend(filtered_docs)

        # 4. Write `all_filtered` back into state["keys"]["documents"]

        # 5. Partition `classified` into answerable vs unanswerable
        answerable = [sq for sq in classified if sq.get("answerable", False)]
        unanswerable = [sq for sq in classified if not sq.get("answerable", False)]

        state.setdefault("keys", {})["documents"] = all_filtered

        state["keys"]["answerable_sub_queries"] = answerable
        state["keys"]["unanswerable_sub_queries"] = unanswerable

        # 6. If any sub-query is unanswerable, signal web search
        # state["keys"]["run_web_search"] = "Yes" if unanswerable else "No"

        end_time = time.time()
        logging.info(
            f"2-step DocumentGrader.run completed at {end_time:.2f} "
            f"(Duration: {end_time - start_time:.2f}s) → "
            f"{len(all_filtered)} total docs, "
            f"{len(answerable)} answerable, "
            f"{len(unanswerable)} unanswerable"
        )

        logging.info(
            "2-Step Document Grader Final State:\n%s",
            json.dumps(state, indent=2, ensure_ascii=False, default=str)
        )
        return state



if __name__ == "__main__":
    # Test the 2-step grading process
    import asyncio

    # Create test data
    docs = [
        {"page_content": "Information about renewable energy benefits...", "metadata": {"score": 600}},
        {"page_content": "Solar panel installation guide...", "metadata": {"score": 800}},  # Auto-include
        {"page_content": "Unrelated content about cooking...", "metadata": {"score": 500}},
    ]
    query = "What are the benefits of renewable energy?"

    # Test synchronous 2-step grading
    grader = DocumentGrader(threshold=1)
    filtered, count, details = grader.grade_documents_two_step(docs, query)
    
    print(f"\n=== 2-Step Grading Test Results ===")
    print(f"Original documents: {len(docs)}")
    print(f"Step 1 relevant: {details['step1_relevant_count']}")
    print(f"Step 2 collective score: {details['step2_collective_score']}")
    print(f"Final documents: {count}")
    print(f"Individual grades: {details['step1_individual_grades']}")
    
    # Test async batch grading
    async def test_batch():
        sub_queries = [{"completed_query": query, "classification": "in-scope"}]
        results = await grader.batch_grade(sub_queries, [query], [docs])
        print(f"\nBatch grading results: {len(results[0][0])} final documents")
        print(f"Grading details: {results[0][2]}")

    asyncio.run(test_batch())