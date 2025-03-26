import logging
from typing import List, Dict
from .document_context_manager import DocumentContextManager
from ...retrieval.retriever_setup import load_ensemble_retriever
from ...config import llama_llm

class ContextAwareRetriever:
    """
    Retrieves documents for each query. Checks if the cached docs in the local file
    are sufficient to answer the query using an LLM-based prompt. If not, performs a new retrieval.
    """

    def __init__(self):
        self.ensemble_retriever = load_ensemble_retriever()  # Your existing ensemble retriever
        self.context_manager = DocumentContextManager()       # Local filestore for doc context

    def run(self, state: dict) -> dict:
        logging.info("Running context-aware retrieval node.")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        session_id = state["keys"].get("session_id")
        if not session_id:
            logging.error("No session_id found in state. Either generate or retrieve one earlier.")
            return state

        if not classified_sub_queries:
            # Fallback: if no sub-queries, just do retrieval for the entire question if needed
            question = state["keys"].get("question", "")
            if not question:
                logging.error("No question or sub-queries provided for retrieval.")
                state["keys"]["documents"] = []
                return state
            
            logging.info(f"Checking if to use cached documents or perform a new retrieval for {session_id} and {question}.")
            # Single retrieval or context check
            docs = self.retrieve_or_use_cache(session_id, question)
            state["keys"]["documents"] = docs
            return state

        # We'll store retrieval results per sub-query
        results = []
        
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")
            logging.info(f"Checking if to use cached documents or perform a new retrieval for {session_id} and {sq_text} which has classification {classification}.")
            if classification == "in-scope":
                docs = self.retrieve_or_use_cache(session_id, sq_text)
                sq_data["documents"] = docs
                results.append(f"Retrieved {len(docs)} docs for in-scope sub-query: {sq_text}")
            else:
                sq_data["documents"] = []
                results.append(f"No retrieval for {classification} sub-query: {sq_text}")

        logging.info("\n".join(results))
        return state

    def retrieve_or_use_cache(self, session_id: str, query: str) -> List[Dict]:
        """
        Decide whether to use the cached docs or do a new retrieval, based on LLM check.
        """
        # 1) Get existing docs from local filestore
        cached_docs_dicts = self.context_manager.get_session_context(session_id)

        # If no cached docs, do retrieval immediately
        if not cached_docs_dicts:
            logging.info("No cached docs found, performing new retrieval.")
            return self.do_new_retrieval(session_id, query, cached_docs_dicts)

        # 2) Combine cached docs into a single context string
        cached_context = self.combine_docs_into_context(cached_docs_dicts)

        # 3) Ask the LLM if the query can be answered from the cached context
        if self.can_answer_from_context(cached_context, query):
            logging.info("LLM indicates the cached docs are sufficient to answer.")
            return cached_docs_dicts
        else:
            logging.info("LLM indicates the cached docs are NOT sufficient, doing new retrieval.")
            return self.do_new_retrieval(session_id, query, cached_docs_dicts)

    def do_new_retrieval(self, session_id: str, query: str, cached_docs_dicts: List[Dict]) -> List[Dict]:
        """Perform a new retrieval, add the new docs to the cache, and return combined docs."""
        logging.info("Performing new retrieval for query: " + query)
        new_docs = self.ensemble_retriever.invoke(query)  # Your retrieval method

        # Convert retrieved docs (LangChain Document) to dict form
        new_docs_dicts = []
        for doc in new_docs:
            new_docs_dicts.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        
        # Add newly retrieved docs to the local filestore
        self.context_manager.add_docs_to_context(session_id, new_docs_dicts)
        
        return new_docs_dicts

    def combine_docs_into_context(self, docs: List[Dict]) -> str:
        """
        Combine multiple doc dicts into a single string context.
        """
        context_texts = []
        for d in docs:
            content_str = d["page_content"]
            context_texts.append(content_str)
        return "\n\n".join(context_texts)

    def can_answer_from_context(self, context_str: str, query: str) -> bool:
        """
        Use an LLM-based approach to see if the query can be confidently answered from context_str alone.
        Return True if yes, False if no.
        """
        qa_prompt = f"""You are a helpful assistant. 
You have the following context:
---
{context_str}
---
User's question: {query}

Step 1: Check if the context is fully sufficient to answer the question.
Step 2: If you can confidently answer from the given context, say "YES". If not, say "NO".
Answer only YES or NO with no explanation.
"""

        try:
            response = llama_llm.invoke(([("system", qa_prompt)]))
            answer = response.content.strip().upper()
            if "YES" in answer:
                return True
            return False
        except Exception as e:
            logging.error(f"Error during LLM-based sufficiency check: {e}")
            return False





# import logging
# from ...config import llama_llm
# from ...retrieval.retriever_setup import load_ensemble_retriever

# class Retriever:
#     """
#     Retrieves documents for each sub-query that is classified as 'in-scope'.
#     """

#     def __init__(self):
#         self.ensemble_retriever = load_ensemble_retriever()

#     def run(self, state: dict) -> dict:
#         logging.info("Running retrieval node.")

#         sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
#         classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

#         if not classified_sub_queries:
#             logging.info("No classified sub-queries found. Falling back to 'question' if available.")
#             # Fallback: if your pipeline sometimes only classifies the entire user query
#             question = state["keys"].get("question", "")
#             if not question:
#                 logging.error("No question or sub-queries provided for retrieval.")
#                 state["keys"]["documents"] = []
#                 return state

#             # Single retrieval for the entire question
#             docs = self.do_retrieval(question)
#             state["keys"]["documents"] = docs
#             return state

#         # We'll store retrieval results per sub-query
#         results = []
#         for sq_data in classified_sub_queries:
#             sq_text = sq_data.get("completed_query", "")
#             classification = sq_data.get("classification", "out-of-scope")

#             if classification == "in-scope":
#                 # Perform retrieval for this sub-query
#                 docs = self.do_retrieval(sq_text)
#                 # Store the docs in the sub-query data
#                 sq_data["documents"] = docs
#                 results.append(f"Retrieved {len(docs)} docs for in-scope sub-query: {sq_text}")
#             else:
#                 # For general or out-of-scope, we skip retrieval
#                 sq_data["documents"] = []
#                 results.append(f"No retrieval for {classification} sub-query: {sq_text}")

#         logging.info("\n".join(results))
#         return state

#     def do_retrieval(self, query_text: str, top_k: int = 8):
#         """
#         Helper method to run retrieval for a single query_text.
#         """
#         if not query_text:
#             return []

#         config = {"configurable": {"search_kwargs_faiss": {"k": top_k}}}
#         try:
#             retrieved_docs = self.ensemble_retriever.invoke(query_text, config=config)
#             return list(retrieved_docs)
#         except Exception as e:
#             logging.error(f"Error during retrieval for '{query_text}': {e}")
#             return []