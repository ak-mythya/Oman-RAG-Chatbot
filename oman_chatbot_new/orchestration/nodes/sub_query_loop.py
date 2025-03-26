import logging
import copy

from .sub_query_identification import SubQueryIdentifier
from .query_classification import QueryClassifier
from .retrieval_node import ContextAwareRetriever
from .document_grader_node import DocumentGrader
from .query_transformer_node import QueryTransformer
from .web_search_node import WebSearch
from .general_out_of_scope import GeneralQueryNode, OutOfScopeQueryNode

class SubQueryLoop:
    """
    Orchestrator that:
      1. Expects sub_query_mapping["sub_queries"] from sub_query_identification
      2. For each sub-query:
         a) Calls query_classification on that single sub-query
         b) If in-scope, calls retrieval_node, doc_grader, etc.
         c) If out-of-scope, calls out_of_scope node
         d) If general, calls general node
      3. Stores results in sub_query_mapping["classified_sub_queries"]
    """

    def __init__(self):
        # Initialize your existing node classes
        self.classifier = QueryClassifier()
        self.retriever = ContextAwareRetriever()
        self.doc_grader = DocumentGrader()
        self.transformer = QueryTransformer()
        self.web_search = WebSearch()
        self.general_node = GeneralQueryNode()
        self.out_of_scope_node = OutOfScopeQueryNode()

    def run(self, state: dict) -> dict:
        logging.info("Running SubQueryLoop node.")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        sub_queries = sub_query_mapping.get("sub_queries", [])
        # If no sub-queries, treat the entire user query as one sub-query
        if not sub_queries:
            logging.info("No sub-queries found. Falling back to the entire user query as one sub-query.")
            user_query = state["keys"].get("question", "")
            if user_query.strip():
                sub_queries = [user_query]
                sub_query_mapping["sub_queries"] = sub_queries
            else:
                logging.info("No user query found either. Skipping.")
                return state


        classified_sub_queries = []

        # Process each sub-query individually
        for sq_text in sub_queries:
            logging.info(f"Processing sub-query: {sq_text}")

            # 1) Create a mini_state so we can classify this sub-query alone
            mini_state = copy.deepcopy(state)
            # Overwrite the sub_query_mapping in mini_state so QueryClassifier sees exactly one sub-query
            mini_state["keys"]["sub_query_mapping"] = {
                "sub_queries": [sq_text]
            }
            # Also set "question" to sq_text so the classifier uses that
            mini_state["keys"]["question"] = sq_text

            # 2) Call the QueryClassifier on this single sub-query
            mini_state = self.classifier.run(mini_state)

            # Now, the classifier stores the result in
            # mini_state["keys"]["sub_query_mapping"]["classified_sub_queries"] = [ { classification: ... } ]
            classified_list = mini_state["keys"]["sub_query_mapping"].get("classified_sub_queries", [])
            if classified_list:
                # We expect exactly one item
                classification = classified_list[0].get("classification", "out-of-scope")
            else:
                classification = "out-of-scope"

            # Prepare data for this sub-query
            sq_data = {
                "completed_query": sq_text,
                "classification": classification,
                "documents": []
            }

            # 3) Handle each classification path
            if classification == "in-scope":
                # (a) Retrieve
                mini_state = self.retriever.run(mini_state)
                docs = mini_state["keys"].get("documents", [])
                sq_data["documents"] = docs

                # (b) Document grader
                mini_state["keys"]["documents"] = docs
                mini_state = self.doc_grader.run(mini_state)
                filtered_docs = mini_state["keys"]["documents"]
                sq_data["documents"] = filtered_docs

                # (c) Possibly decide next step (transform query or web search) if doc grader says so
                run_web_search = mini_state["keys"].get("run_web_search", "No")
                if run_web_search == "Yes":
                    mini_state = self.transformer.run(mini_state)
                    mini_state = self.web_search.run(mini_state)
                    # Merge new docs
                    sq_data["documents"] = mini_state["keys"]["documents"]

            elif classification == "general":
                # handle with general node
                mini_state["keys"]["question"] = sq_text
                mini_state = self.general_node.run(mini_state)
                # For demonstration, we store a placeholder or parse from mini_state
                sq_data["response"] = "some general response"

            else:  # out-of-scope
                mini_state["keys"]["question"] = sq_text
                mini_state = self.out_of_scope_node.run(mini_state)
                sq_data["response"] = "some out-of-scope fallback"

            classified_sub_queries.append(sq_data)

        # 4) Store final sub-query data
        sub_query_mapping["classified_sub_queries"] = classified_sub_queries
        state["keys"]["sub_query_mapping"] = sub_query_mapping

        return state
