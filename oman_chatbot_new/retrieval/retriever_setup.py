"""
Module: retriever_setup.py
This module sets up the EnsembleRetriever using a FAISS index and BM25 retriever.
It uses a RetrieverManager class to ingest documents and build (or load) the vector index.
"""

import os
import sys
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
from oman_chatbot_new.config import DATA_PATH, DB_FAISS_PATH, embeddings, ensemble_retriever_global
from oman_chatbot_new.data_ingestion.data_ingestion_pipeline import DocumentIngestionPipeline

LANGCHAIN_DOCS_PATH = "[AI]langchain_docs.json"


class RetrieverManager:
    """
    Manages document ingestion, vector index creation/loading, and EnsembleRetriever setup.
    """

    def __init__(self):
        self.ensemble_retriever = None
        current_dir = Path(__file__).resolve().parent
        # e.g., C:\Anmol\Oman Chatbot\oman_chatbot_new\retrieval
        # Go one level up to get "oman_chatbot_new", then into data_ingestion/data/Report.pdf
        self.pdf_files = [current_dir.parent / "data_ingestion" / "data" / "eval_pdf.pdf"]
        self.md_output_dir = current_dir.parent / "converted_markdown"
        self.ingestion_pipeline = DocumentIngestionPipeline(input_paths=self.pdf_files, output_dir=self.md_output_dir, clean=True, caption=False)

    def save_langchain_docs(self, docs, file_path: str) -> None:
        """Save LangChain documents as a JSON file to avoid recomputation."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([doc.dict() for doc in docs], f)

    def load_langchain_docs(self, file_path: str):
        """Load LangChain documents from a JSON file if it exists."""
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Document(**doc) for doc in data]

    def load_retriever(self):
        """
        Loads a FAISS vector store if available; otherwise, runs full ingestion.
        Then creates an EnsembleRetriever (combining vector and BM25 retrievers).
        """
        base_dir = Path(__file__).resolve().parent
        pdf_path = base_dir.parent / "data_ingestion" / "data" / "eval_pdf.pdf"
        output_dir = Path("converted_markdown")

        # Try loading preprocessed LangChain documents
        langchain_docs = self.load_langchain_docs(LANGCHAIN_DOCS_PATH)

        if Path(DB_FAISS_PATH).exists():
            print("Loading existing FAISS index...")
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            # If LangChain docs are missing (needed for BM25), re-create them
            if not langchain_docs:
                print("Regenerating LangChain documents for BM25...")
                langchain_docs = self.ingestion_pipeline.create_langchain_documents()
                self.save_langchain_docs(langchain_docs, LANGCHAIN_DOCS_PATH)
        else:
            print("No FAISS index found. Running full ingestion pipeline...")
            langchain_docs = self.ingestion_pipeline.create_langchain_documents()
            self.save_langchain_docs(langchain_docs, LANGCHAIN_DOCS_PATH)
            db = FAISS.from_documents(langchain_docs, embeddings)
            db.save_local(DB_FAISS_PATH)  # Persist the FAISS index

        # Initialize retrievers
        vectorstore_retriever = db.as_retriever(search_kwargs={"k": 8})
        keyword_retriever = BM25Retriever.from_documents(langchain_docs)

        # Create the ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )

        print("EnsembleRetriever is ready.")
        return self.ensemble_retriever


def load_ensemble_retriever():
    """
    Global function to load the ensemble retriever.
    """
    global ensemble_retriever_global
    manager = RetrieverManager()
    ensemble_retriever_global = manager.load_retriever()
    return ensemble_retriever_global
