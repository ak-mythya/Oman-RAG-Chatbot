import json
import os
import logging
from pathlib import Path
from typing import List, Dict

class DocumentContextManager:
    """
    Manages cached context (retrieved documents) for each session,
    storing them in a local JSON file. This allows you to skip retrieval
    if the context is already sufficient to answer a new query.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.context_file = current_dir.parent.parent / "context_store.json"
        self.context_store = self.load_context_store()

    def load_context_store(self):
        if os.path.exists(self.context_file):
            if os.path.getsize(self.context_file) == 0:
                # File is empty, return an empty dict
                logging.warning(f"{self.context_file} is empty; returning empty context store.")
                return {}
            try:
                with open(self.context_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading context store: {e}")
                return {}
        return {}


    def save_context_store(self):
        """Saves the current context store to a file."""
        try:
            with open(self.context_file, "w", encoding="utf-8") as f:
                json.dump(self.context_store, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving context store: {e}")

    def get_session_context(self, session_id: str) -> List[Dict]:
        """
        Retrieve the context (list of documents as dicts) for a given session ID.
        Each doc dict has the form:
        {
          "page_content": "...",
          "metadata": {...}
        }
        """
        return self.context_store.get(session_id, [])

    def add_docs_to_context(self, session_id: str, docs: List[Dict]):
        """
        Add new docs to the session's context. Each doc is a dict with:
          {
            "page_content": "...",
            "metadata": {...}
          }
        """
        existing_docs = self.context_store.get(session_id, [])
        existing_docs.extend(docs)
        self.context_store[session_id] = existing_docs
        self.save_context_store()

    def clear_context(self, session_id: str):
        """Remove all context for a given session ID."""
        if session_id in self.context_store:
            del self.context_store[session_id]
            self.save_context_store()
