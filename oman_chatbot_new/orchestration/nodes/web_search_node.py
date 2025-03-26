import logging
from ...config import TAVILY_API_KEY
from langchain.schema import Document
from tavily import TavilyClient

class WebSearch:
    """
    Uses the Tavily API to perform a web search and append the results as documents.
    This now handles multiple sub-queries, searching only for in-scope sub-queries 
    where the `did_transform` flag is True.
    """

    def __init__(self, tavily_api_key: str = TAVILY_API_KEY):
        self.tavily_api_key = tavily_api_key

    def run(self, state: dict) -> dict:
        logging.info("Running web search node.")

        # Retrieve sub-query mapping and classified sub-queries
        sub_query_mapping = state.setdefault("keys", {}).get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
        documents = state.get("keys", {}).get("documents", [])

        if not self.tavily_api_key:
            logging.info("No Tavily API key provided - skipping web search.")
            state.setdefault("keys", {})["documents"] = documents
            return state

        # Iterate over each classified sub-query
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")

            # Check if the sub-query is 'in-scope' and has been transformed
            if classification == "in-scope" and sq_data.get("did_transform", False):
                query_to_search = sq_data.get("transformed_query", sq_text)  # Use the transformed query if available
                try:
                    # Perform the search only for in-scope queries that are transformed
                    tavily_client = TavilyClient(api_key=self.tavily_api_key)
                    response = tavily_client.search(query_to_search)
                    web_results = []

                    # Collect results and format them into documents
                    for result in response.get("results", []):
                        content = f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\n"
                        web_results.append(content)
                    if web_results:
                        web_document = Document(
                            page_content="\n\n".join(web_results),
                            metadata={"source": "tavily_search", "query": query_to_search, "result_count": len(web_results)}
                        )
                        sq_data["documents"] = web_document  # Add the search result to the sub-query data
                        logging.info(f"Successfully added {len(web_results)} web results for sub-query: {query_to_search}")
                    else:
                        sq_data["documents"] = []  # No results found

                except Exception as e:
                    logging.error(f"Error performing web search for sub-query '{sq_text}': {e}")
                    sq_data["documents"] = []  # Set documents to empty in case of an error

        # Update state with all sub-query documents
        state.setdefault("keys", {})["documents"] = documents
        return state
