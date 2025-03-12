import logging
from ...config import TAVILY_API_KEY
from langchain.schema import Document
from tavily import TavilyClient

class WebSearch:
    """
    Uses the Tavily API to perform a web search and append the results as documents.
    """

    def __init__(self, tavily_api_key: str = TAVILY_API_KEY):
        self.tavily_api_key = tavily_api_key

    def run(self, state: dict) -> dict:
        logging.info("Running web search node.")
        question = state.get("keys", {}).get("question", "")
        documents = state.get("keys", {}).get("documents", [])

        if not self.tavily_api_key:
            logging.info("No Tavily API key provided - skipping web search.")
            state.setdefault("keys", {})["documents"] = documents
            return state

        try:
            tavily_client = TavilyClient(api_key=self.tavily_api_key)
            response = tavily_client.search(question)
            web_results = []
            for result in response.get("results", []):
                content = f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\n"
                web_results.append(content)
            if web_results:
                web_document = Document(
                    page_content="\n\n".join(web_results),
                    metadata={"source": "tavily_search", "query": question, "result_count": len(web_results)}
                )
                documents.append(web_document)
                logging.info(f"Successfully added {len(web_results)} web results.")
        except Exception as e:
            logging.error(f"Web search error: {e}")

        state.setdefault("keys", {})["documents"] = documents
        return state
