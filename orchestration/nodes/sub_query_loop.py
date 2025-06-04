import logging
import copy
import asyncio
import time
import json
from .sub_query_identification import SubQueryIdentifier
from .query_classification import QueryClassifier
from .retrieval_node import ContextAwareRetriever
from .document_grader_node import DocumentGrader
from .query_transformer_node import QueryTransformer
from .web_search_node import WebSearch
from .general_out_of_scope import GeneralQueryNode

class SubQueryLoop:
    """Handles processing of sub-queries through classification, retrieval, grading, and response generation."""
    
    def __init__(self):
        """Initialize the SubQueryLoop with necessary components."""
        self.classifier = QueryClassifier()
        self.retriever = ContextAwareRetriever()
        self.doc_grader = DocumentGrader()
        self.transformer = QueryTransformer()
        self.web_search = WebSearch()
        self.general_node = GeneralQueryNode()
        self.logger = logging.getLogger(__name__)

    async def _process_in_scope_query(self, sub_query: dict, state: dict, docs: list[dict], relevant_count: int) -> dict:
        """Process an in-scope sub-query with retrieved and graded documents."""
        sq_data = {
            "completed_query": sub_query["completed_query"],
            "classification": sub_query["classification"],
            "documents": docs,
            "relevant_count": relevant_count,
            # carry the answerable flag forward
            "answerable": sub_query.get("answerable", False)
        }

        mini_state = copy.deepcopy(state)
        mini_state["keys"]["question"] = sub_query["completed_query"]
        mini_state["keys"]["sub_query_mapping"] = {
            "classified_sub_queries": [sub_query]
        }
        mini_state["keys"]["documents"] = docs
        mini_state["keys"]["relevant_count"] = relevant_count

        return sq_data

    async def process_classified_sub_query(self, sub_query: dict, state: dict, docs: list[dict] = None, relevant_count: int = 0) -> dict:
        """Process a single classified sub-query based on its classification."""
        start_time = time.time()
        sq_text = sub_query["completed_query"]
        classification = sub_query["classification"]
        self.logger.info(f"Processing sub-query: '{sq_text}' (Classification: {classification})")

        try:
            if classification == "in-scope":
                result = await self._process_in_scope_query(sub_query, state, docs or [], relevant_count)
            elif classification == "human-agent":
                result = {
                    "completed_query": sq_text,
                    "classification": classification,
                    "response": f"Please contact a human agent for assistance with '{sq_text}'."
                }
            elif classification == "service-application":
                result = {
                    "completed_query": sq_text,
                    "classification": classification,
                    "response": f"Please use the service application for '{sq_text}'."
                }
            elif classification == "child-abuse":
                result = {
                    "completed_query": sq_text,
                    "classification": classification,
                    "response": "يرجى العثور على رابط نموذج إساءة معاملة الأطفال: https://portal.mosd.gov.om/webcenter/portal/MOSDExternalPortal/pages_services/reportabuse"
                }
            elif classification == "out-of-scope":
                result = {
                    "completed_query": sq_text,
                    "classification": classification,
                    "response": "نعتذر، الموضوع خارج اختصاص عمل الوزارة"
                }
            else:  # e.g., "exit-chat" or unexpected classifications
                result = {
                    "completed_query": sq_text,
                    "classification": classification,
                    "response": "exit-chat"
                }

            duration = time.time() - start_time
            self.logger.info(f"Completed sub-query: '{sq_text}' in {duration:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Error processing sub-query '{sq_text}': {str(e)}")
            return {
                "completed_query": sq_text,
                "classification": classification,
                "response": f"Error: {str(e)}"
            }

    async def _batch_classify_sub_queries(self, state: dict, sub_queries: list[str]) -> list[dict]:
        """Classify all sub-queries in a batch."""
        start_time = time.time()
        self.logger.info(f"Starting batch classification for {len(sub_queries)} sub-queries")

        classification_state = copy.deepcopy(state)
        classification_state["keys"]["sub_query_mapping"] = {"sub_queries": sub_queries}
        classification_state = await self.classifier.run(classification_state)
        classified_sub_queries = classification_state["keys"]["sub_query_mapping"].get("classified_sub_queries", [])

        duration = time.time() - start_time
        self.logger.info(f"Completed batch classification in {duration:.2f} seconds")
        return classified_sub_queries


    async def _batch_retrieve_and_grade(
        self,
        state: dict,
        in_scope_sub_queries: list[dict],
        in_scope_queries: list[str],
    ) -> list[tuple[list[dict], int]]:
        """Perform batch retrieval and grading for in-scope sub-queries, tag each as answerable or not, and update state."""
        session_id = state["keys"].get("session_id", "")
        if not session_id:
            self.logger.error("No session_id found in state")
            return []

        # 1) Retrieve
        start_time = time.time()
        self.logger.info(f"Starting batch retrieval for {len(in_scope_queries)} sub-queries")
        retrieved_docs = await self.retriever.batch_retrieve(session_id, in_scope_queries)
        duration = time.time() - start_time
        self.logger.info(f"Completed batch retrieval in {duration:.2f} seconds")

        # 2) Grade
        start_time = time.time()
        self.logger.info(f"Starting batch grading for {len(in_scope_queries)} sub-queries")
        graded_results = await self.doc_grader.batch_grade(
            in_scope_sub_queries,
            in_scope_queries,
            retrieved_docs
        )
        self.logger.info(f"Document Grader Node graded results: {graded_results}")
        duration = time.time() - start_time
        self.logger.info(f"Completed batch grading in {duration:.2f} seconds")

        # 3) Update each sub-query with filtered docs, count, and answerable flag
        all_filtered_docs = []
        new_results = []
        for (filtered_docs, count, score), sq_data in zip(graded_results, in_scope_sub_queries):
            sq_data["documents"]      = filtered_docs
            sq_data["relevant_count"] = count
            # Tag as answerable if we met the threshold
            sq_data["answerable"]     = score == "yes"

            all_filtered_docs.extend(filtered_docs)
            new_results.append((filtered_docs, count))

        # 4) Persist answerable/unanswerable lists into state
        keys = state.setdefault("keys", {})
        keys["answerable_sub_queries"]   = [
            sq for sq in in_scope_sub_queries if sq.get("answerable")
        ]
        keys["unanswerable_sub_queries"] = [
            sq for sq in in_scope_sub_queries if not sq.get("answerable")
        ]

        #(Optional) keep the flattened docs for the answerable ones
        keys["documents"] = [
            doc for sq in keys["answerable_sub_queries"] for doc in sq["documents"]
        ]
        self.logger.info(
    "ANSWERABLE SUBQUERIES: %s",
    [sq["completed_query"] for sq in keys["answerable_sub_queries"]]
)
        #keys["documents"] = all_filtered_docs

        end_time = time.time()
        self.logger.info(
            f"Finished _batch_retrieve_and_grade at {end_time:.2f} "
            f"(Duration: {end_time - start_time:.2f} seconds)"
        )
        return new_results

    async def run(self, state: dict) -> dict:
        """Run the sub-query processing loop with enhanced handling for out-of-scope queries."""
        start_time = time.time()
        self.logger.info("Starting SubQueryLoop.run")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        sub_queries = sub_query_mapping.get("sub_queries", [])

        if not sub_queries:
            self.logger.info("No sub-queries found. Using user query as fallback.")
            user_query = state["keys"].get("question", "").strip()
            if not user_query:
                self.logger.info("No user query found. Exiting.")
                return state
            sub_queries = [user_query]
            sub_query_mapping["sub_queries"] = sub_queries

        # Classify sub-queries
        classified_sub_queries = await self._batch_classify_sub_queries(state, sub_queries)
        sub_query_mapping["classified_sub_queries"] = classified_sub_queries
        # ─── NEW: detect if all sub-queries are classified "general" ───
        only_general = bool(classified_sub_queries) and all(
            sq.get("classification") == "general"
            for sq in classified_sub_queries
        )
        state["keys"]["only_general_request"] = only_general
        # ─── ADD THIS ───  
        escalation_classes = {"out-of-scope", "human-agent", "exit-chat", "service-application", "child-abuse"}

        # Determine if there are multiple sub-queries with no in-scope queries
        classifications = [sq.get("classification") for sq in classified_sub_queries]
        has_in_scope = any(c == "in-scope" for c in classifications)
        all_out_of_scope = bool(classified_sub_queries) and all(
            sq.get("classification") == "out-of-scope" for sq in classified_sub_queries
        )
        multiple_no_inscope = len(sub_queries) > 1 and not has_in_scope and not all_out_of_scope
        state["keys"]["multiple_no_inscope"] = multiple_no_inscope
        
        has_hard = any(c in escalation_classes for c in classifications)
        has_in_scope = any(c == "in-scope" for c in classifications)

        if has_hard and not has_in_scope:
            # gather them in order, dedupe
            found = [c for c in classifications if c in escalation_classes]
            state["keys"]["escalation_classes"]   = list(dict.fromkeys(found))
            state["keys"]["has_escalation_request"] = True
        else:
            state["keys"]["escalation_classes"]   = []
            state["keys"]["has_escalation_request"] = False


        # Separate queries by type
        in_scope_sub_queries = [sq for sq in classified_sub_queries if sq.get("classification") == "in-scope"]
        general_sub_queries = [sq for sq in classified_sub_queries if sq.get("classification") == "general"]
        in_scope_queries = [sq["completed_query"] for sq in in_scope_sub_queries]

        # Perform batch retrieval and grading for in-scope queries
        graded_results = await self._batch_retrieve_and_grade(state, in_scope_sub_queries, in_scope_queries) if in_scope_queries else []
         # ─── NEW: flag when there's exactly one in-scope query (and nothing else) ───
        # only_one = (
        #     len(in_scope_sub_queries) == 1
        #     # and len(general_sub_queries) == 0
        #     # and len(out_of_scope_sub_queries) == 0
        # )
        # state["keys"]["only_one_in_scope"] = only_one
        # only true if the entire list is exactly ["in-scope"]
        only_one = (classifications == ["in-scope"])
        state["keys"]["only_one_in_scope"] = only_one

        # Process general queries in batch
        if not state["keys"].get("only_general_request", False) and general_sub_queries:
            self.logger.info(f"Processing {len(general_sub_queries)} general sub-queries in batch")
            mini_state = copy.deepcopy(state)
            mini_state["keys"]["sub_query_mapping"] = {"classified_sub_queries": general_sub_queries}
            mini_state = await self.general_node.run(mini_state)
            for sq, updated_sq in zip(general_sub_queries, mini_state["keys"]["sub_query_mapping"]["classified_sub_queries"]):
                sq["response"] = updated_sq.get("response", "No response generated.")

        # Process all non-general sub-queries (including out-of-scope)
        tasks = []
        for sq in classified_sub_queries:
            if sq.get("classification") != "general":
                if sq.get("classification") == "in-scope":
                    try:
                        idx = in_scope_sub_queries.index(sq)
                        graded_docs, relevant_count = graded_results[idx]
                        tasks.append(self.process_classified_sub_query(sq, state, graded_docs, relevant_count))
                    except ValueError:
                        self.logger.error(f"In-scope query '{sq['completed_query']}' not found in graded_results")
                        tasks.append(self.process_classified_sub_query(sq, state))
                else:
                    tasks.append(self.process_classified_sub_query(sq, state))

        # Execute tasks concurrently
        start_time = time.time()
        self.logger.info("Starting batch processing of non-general sub-queries")
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = list(processed_results) + general_sub_queries
        duration = time.time() - start_time
        self.logger.info(f"Completed batch processing in {duration:.2f} seconds")

        # Update state with results
        sub_query_mapping["classified_sub_queries"] = processed_results
        sub_query_mapping["sub_query_answers"] = [
            {"completed_query": sq["completed_query"], "response": sq.get("response")}
            for sq in processed_results if "response" in sq
        ]

        # Aggregate documents
        all_filtered_docs = [doc for sq in processed_results for doc in sq.get("documents", [])]
        state["keys"]["documents"] = all_filtered_docs

        duration = time.time() - start_time
        self.logger.info(f"Completed SubQueryLoop.run in {duration:.2f} seconds")
        return state