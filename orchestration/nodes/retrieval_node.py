import logging
import asyncio
import time
from typing import List, Dict
from retrieval.retriever_setup import load_ensemble_retriever
from model import TextGenerator

class ContextAwareRetriever:
    """
    Retrieves documents for multiple sub-queries in a batch. Checks if cached docs in the local file
    are sufficient to answer each query using an LLM-based prompt. If not, performs a new retrieval.
    Uses async batch processing and logging to verify concurrency.
    """

    def __init__(self):
        self.ensemble_retriever = load_ensemble_retriever()  # Updated to use FAISS via RetrieverManager
        self.generator = TextGenerator()

    async def process_sub_query(self, sq_data: Dict, session_id: str, docs: List[Dict]) -> None:
        """
        Process a single sub-query by assigning retrieved documents.
        """
        sq_text = sq_data.get("completed_query", "")
        classification = sq_data.get("classification", "out-of-scope")
        start = time.time()
        logging.info(f"[Retriever] START processing '{sq_text}' at {start:.3f}")

        if classification == "in-scope":
            sq_data["documents"] = docs
            logging.info(f"[Retriever] Assigned {len(docs)} docs for in-scope sub-query: {sq_text}")
        else:
            sq_data["documents"] = []
            logging.info(f"[Retriever] No retrieval for {classification} sub-query: {sq_text}")

        end = time.time()
        logging.info(f"[Retriever] END   processing '{sq_text}' at {end:.3f} (took {end-start:.3f}s)")

    async def batch_retrieve(self, session_id: str, queries: List[str]) -> List[List[Dict]]:
        """
        Perform batched retrieval for a list of queries concurrently.
        """
        start_time = time.time()
        logging.info(f"[Retriever] Started batch_retrieve for {len(queries)} queries at {start_time:.2f}")

        async def retrieve_single_query(query: str) -> List[Dict]:
            retrieve_start = time.time()
            logging.info(f"[Retriever] Started retrieval for query '{query}' at {retrieve_start:.2f}")
            docs = await asyncio.to_thread(self.retrieve_or_use_cache, session_id, query)
            retrieve_end = time.time()
            logging.info(f"[Retriever] Finished retrieval for query '{query}' at {retrieve_end:.2f} (Duration: {retrieve_end - retrieve_start:.2f} seconds)")
            return docs

        tasks = [retrieve_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logging.error(f"[Retriever] Failed to retrieve for query '{query}': {result}")
                results[results.index(result)] = []

        end_time = time.time()
        logging.info(f"[Retriever] Finished batch_retrieve for {len(queries)} queries at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
        return results

    async def run(self, state: dict) -> dict:
        """
        Run retrieval for all sub-queries or fallback to single query.
        """
        start_time = time.time()
        logging.info(f"[Retriever] Started ContextAwareRetriever.run at {start_time:.2f}")

        sub_query_mapping = state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
        classified = sub_query_mapping.get("classified_sub_queries", [])

        session_id = state["keys"].get("session_id")
        if not session_id:
            logging.error("No session_id found in state. Either generate or retrieve one earlier.")
            end_time = time.time()
            logging.info(f"[Retriever] Finished ContextAwareRetriever.run at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
            return state

        if not classified:
            question = state["keys"].get("question", "")
            if not question:
                logging.error("No question or sub-queries provided for retrieval.")
                state["keys"]["documents"] = []
                end_time = time.time()
                logging.info(f"[Retriever] Finished ContextAwareRetriever.run at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
                return state

            logging.info(f"[Retriever] Starting fallback retrieval for '{question}' at {time.time():.2f}")
            docs = await asyncio.to_thread(self.retrieve_or_use_cache, session_id, question)
            state["keys"]["documents"] = docs
            end_time = time.time()
            logging.info(f"[Retriever] Finished ContextAwareRetriever.run at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
            return state

        in_scope_queries = []
        in_scope_sub_queries = []
        for sq in classified:
            if sq.get("classification") == "in-scope":
                query = sq.get("completed_query", "")
                in_scope_queries.append(query)
                in_scope_sub_queries.append(sq)

        if in_scope_queries:
            retrieved_docs = await self.batch_retrieve(session_id, in_scope_queries)
            tasks = [
                self.process_sub_query(sq, session_id, docs)
                for sq, docs in zip(in_scope_sub_queries, retrieved_docs)
            ]
            await asyncio.gather(*tasks)
        else:
            tasks = [self.process_sub_query(sq, session_id, []) for sq in classified if sq.get("classification") != "in-scope"]
            if tasks:
                await asyncio.gather(*tasks)

        end_time = time.time()
        logging.info(f"[Retriever] Finished ContextAwareRetriever.run at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
        return state

    def retrieve_or_use_cache(self, session_id: str, query: str) -> List[Dict]:
        """
        Perform retrieval directly (cache logic commented out as in original).
        """
        return self.do_new_retrieval(session_id, query, [])

    def do_new_retrieval(self, session_id: str, query: str, cached: List[Dict]) -> List[Dict]:
        start_time = time.time()
        logging.info(f"[Retriever] Started new retrieval for query '{query}' at {start_time:.2f}")
        new_docs = self.ensemble_retriever.invoke(query)
        new_dicts = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in new_docs
        ]
        end_time = time.time()
        logging.info(f"retrieved_docs: {new_dicts}")
        logging.info(f"[Retriever] Finished new retrieval for query '{query}' at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
        return new_dicts

    def combine_docs_into_context(self, docs: List[Dict]) -> str:
        start_time = time.time()
        logging.info(f"[Retriever] Started combining docs at {start_time:.2f}")
        context = "\n\n".join(d.get("page_content", "") for d in docs)
        end_time = time.time()
        logging.info(f"[Retriever] Finished combining docs at {end_time:.2f} (Duration: {end_time - start_time:.2f} seconds)")
        return context

    def can_answer_from_context(self, context_str: str, query: str) -> bool:
        return False