import os
os.environ["TRANSFORMERS_CACHE"] = "E:/Tejas Taneja/huggingface_user2"
os.environ["HF_DATASETS_CACHE"] = "E:/Tejas Taneja/huggingface_user2"

import logging
from functools import wraps
import time
from typing import List
from redisvl.extensions.llmcache import SemanticCache
from config import cache_emb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def time_logger(func):
    """
    Decorator to log the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Function '{func.__name__}' took {duration:.4f} seconds to execute")
        return result
    return wrapper

class DocumentContextManager:
    @time_logger
    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.1,
        ttl_seconds: int = 3600,
    ):
        """
        Initialize the DocumentContextManager with a SemanticCache instance.

        Args:
            session_id (str): Unique identifier for the session (used as cache name).
            redis_url (str): Redis connection URL.
            distance_threshold (float): Similarity threshold for cache matching.
            ttl_seconds (int): Time-to-live for cache entries in seconds.
        """
        try:
            self.llmcache = SemanticCache(
                name=session_id,
                redis_url=redis_url,
                distance_threshold=distance_threshold,
                vectorizer=cache_emb,
                ttl=ttl_seconds,
                overwrite=False,
            )
            logger.info(f"Successfully initialized SemanticCache with name '{session_id}'.")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticCache: {e}")
            raise

    @time_logger
    def add_docs_to_context(self, prompt: str, response: str) -> None:
        """
        Add a response to the cache, using the response for both vectorization and storage.

        Args:
            response (str): The response to cache and use for similarity search.
        """
        try:
            self.llmcache.store(prompt=prompt, response=response)
            logger.info(f"Cache set for response: {response[:50]}...")
        except Exception as e:
            logger.error(f"Error storing cache for response '{response[:50]}...': {e}")
            raise

    @time_logger
    def get_session_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve cached responses for a given query based on semantic similarity.

        Args:
            query (str): The query to check in the cache.
            top_k (int): The maximum number of results to return.

        Returns:
            List[str]: A list of cached responses, or an empty list if no cache hit.
        """
        try:
            cached_docs = self.llmcache.check(prompt=query, num_results=top_k, return_fields=['response'])
            if cached_docs:
                logger.info(f"Cache hit for query: {query}")
                return [result['response'] for result in cached_docs]
            logger.info(f"Cache miss for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving cache for query '{query}': {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Create a manager for a specific session
    mgr = DocumentContextManager(session_id="user-1234")

    # Add a response to the cache
    prompt = response = "Oman is on the southeastern coast of the Arabian Peninsula."
    mgr.add_docs_to_context(prompt=prompt, response=response)

    # Retrieve cached responses for a similar query
    retrieved = mgr.get_session_context(query="Which country is in Arabian Peninsula?", top_k=2)
    print("Retrieved:", retrieved)