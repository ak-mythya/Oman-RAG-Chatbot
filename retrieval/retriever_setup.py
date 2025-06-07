# import os
# import json
# from pathlib import Path
# import logging
# from langchain.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.docstore.document import Document
# from config import embeddings
# from data_ingestion.data_ingestion_pipeline import DocumentIngestionPipeline

# logger = logging.getLogger(__name__)

# # File paths
# LANGCHAIN_DOCS_PATH = "E:/Tejas Taneja/Oman-RAG-Chatbot-Main/oman_chatbot_new/arabic_chunks.json"
# FAISS_INDEX_PATH = "E:/Tejas Taneja/Oman-RAG-Chatbot-Main/oman_chatbot_new/faiss_index"

# # Global vector store (initially None)
# vector_store = None

# class VectorStoreRetrieverWithScore:
#     def __init__(self, vectorstore, search_kwargs):
#         self.vectorstore = vectorstore
#         self.search_kwargs = search_kwargs

#     def get_relevant_documents(self, query: str) -> list[Document]:
#         docs_and_scores = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
#         for doc, score in docs_and_scores:
#             doc.metadata['score'] = score
#         return [doc for doc, score in docs_and_scores]

# class RetrieverManager:
#     """
#     Manages document ingestion, FAISS index creation/loading, and EnsembleRetriever setup.
#     """

#     def __init__(self):
#         current_dir = Path(__file__).resolve().parent
#         self.pdf_files_dir = current_dir.parent / "data_ingestion" / "data"
#         self.md_output_dir = current_dir.parent / "converted_markdown"
#         self.ingestion_pipeline = DocumentIngestionPipeline(
#             input_paths=self.pdf_files_dir,
#             output_dir=self.md_output_dir,
#             clean=True
#         )
#         self.ensemble_retriever = None

#     def save_langchain_docs(self, docs, file_path: str) -> None:
#         """Save LangChain documents as a JSON file."""
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump([doc.dict() for doc in docs], f)

#     def load_langchain_docs(self, file_path: str):
#         """Load LangChain documents from a JSON file if it exists."""
#         if not os.path.exists(file_path):
#             return None
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             return [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in data]

#     def load_retriever(self):
#         global vector_store
#         """
#         Ingest documents (if needed), create or load a FAISS index,
#         and set up the EnsembleRetriever using a FAISS vector retriever and a BM25 retriever.
#         """
#         print(f"Checking for preprocessed docs at {LANGCHAIN_DOCS_PATH}...")
#         langchain_docs = self.load_langchain_docs(LANGCHAIN_DOCS_PATH)
#         if not langchain_docs:
#             print(f"No preprocessed docs found at {LANGCHAIN_DOCS_PATH}. Running ingestion pipeline...")
#             langchain_docs = self.ingestion_pipeline.create_langchain_documents()
#             self.save_langchain_docs(langchain_docs, LANGCHAIN_DOCS_PATH)

#         if not os.path.exists(FAISS_INDEX_PATH):
#             print(f"FAISS index not found at {FAISS_INDEX_PATH}. Building it...")
#             vector_store = FAISS.from_documents(
#                 documents=langchain_docs,
#                 embedding=embeddings,
#                 distance_strategy=DistanceStrategy.COSINE
#             )
#             vector_store.save_local(FAISS_INDEX_PATH)
#         else:
#             print(f"FAISS index found at {FAISS_INDEX_PATH}. Loading it...")
#             vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

#         vectorstore_retriever = VectorStoreRetrieverWithScore(
#             vectorstore=vector_store,
#             search_kwargs={"k": 3}
#         )
#         keyword_retriever = BM25Retriever.from_documents(langchain_docs, k1=1.5, b=0.75)
#         self.ensemble_retriever = EnsembleRetriever(
#             retrievers=[vectorstore_retriever, keyword_retriever],
#             weights=[0.5, 0.5],
#         )
#         print("EnsembleRetriever is ready.")
#         return self.ensemble_retriever

# ensemble_retriever_global = None

# def load_ensemble_retriever():
#     global ensemble_retriever_global, vector_store
#     if ensemble_retriever_global is None:
#         print("Loading ensemble retriever...")
#         manager = RetrieverManager()
#         ensemble_retriever_global = manager.load_retriever()
#     else:
#         print("Ensemble retriever already loaded.")
#     return ensemble_retriever_global

import os
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from uuid import uuid4

import faiss
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from config import embeddings
from config import DATA_PATH, ensemble_retriever_global
from data_ingestion.data_ingestion_pipeline import DocumentIngestionPipeline

logger = logging.getLogger(__name__)

# ---------------------------
# FAISS index & cache settings
# ---------------------------
FAISS_INDEX_DIR = "faiss_index"
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # oman_chatbot_new/
LANGCHAIN_DOCS_PATH = BASE_DIR / "arabic_chunks.json" 

# Global in‑memory vector store reference
vector_store = None

# class VectorStoreRetrieverWithScore(VectorStoreRetriever):
#     """Wraps a LangChain VectorStoreRetriever to attach similarity scores to metadata"""

#     def get_relevant_documents(self, query: str) -> List[Document]:
#         docs_and_scores = self.vectorstore.similarity_search_with_score(
#             query, **self.search_kwargs
#         )
#         for doc, score in docs_and_scores:
#             doc.metadata['score'] = score
#         return [doc for doc, _ in docs_and_scores]

class VectorStoreRetrieverWithScore(VectorStoreRetriever):
    """Wraps a LangChain VectorStoreRetriever to attach similarity scores to metadata"""
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs
        )
        for doc, score in docs_and_scores:
            # Convert numpy.float32 to Python float
            doc.metadata['score'] = float(score)
        return [doc for doc, _ in docs_and_scores]


class RetrieverManager:
    """
    Manages:
      - PDF→Markdown ingestion
      - LangChain doc JSON caching
      - FAISS index creation/loading
      - BM25 + FAISS EnsembleRetriever
    """

    def __init__(self):
        base = Path(__file__).resolve().parent
        self.pdf_files_dir = base.parent / "data_ingestion" / "data"
        self.md_output_dir = base.parent / "converted_markdown"
        self.ingestion_pipeline = DocumentIngestionPipeline(
            input_paths=self.pdf_files_dir,
            output_dir=self.md_output_dir,
            clean=True
        )
        self.ensemble_retriever = None

    def save_langchain_docs(self, docs: List[Document], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([{"text": d.page_content, "metadata": d.metadata} for d in docs], f)

    def load_langchain_docs(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]

    def build_faiss_index(self, docs: List[Document]) -> FAISS:
        """Creates a new FAISS index from documents and saves it to disk."""
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=embeddings,
            normalize_L2=True
        )
        # index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))

        # vector_store = FAISS(
        #     embedding_function=embeddings,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},
        # )
        # uuids = [str(uuid4()) for _ in range(len(docs))]

        # vector_store.add_documents(documents=docs, ids=uuids)  
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vector_store.save_local(FAISS_INDEX_DIR)
        return vector_store

    def load_retriever(self) -> EnsembleRetriever:
        global vector_store

        # 1. Load or ingest LangChain docs
        langchain_docs = self.load_langchain_docs(LANGCHAIN_DOCS_PATH)
        if not langchain_docs:
            logger.info("No preprocessed docs found; running ingestion pipeline...")
            langchain_docs = self.ingestion_pipeline.create_langchain_documents()
            self.save_langchain_docs(langchain_docs, LANGCHAIN_DOCS_PATH)
        else:
            logger.info(f"Loaded {len(langchain_docs)} cached LangChain docs.")

        # 2. Build or load FAISS vector store
        if not os.path.exists(FAISS_INDEX_DIR):
            logger.info("FAISS index not found; building new index...")
            vector_store = self.build_faiss_index(langchain_docs)
        else:
            logger.info("Loading existing FAISS index from disk...")
            vector_store = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )

        # 3. Wrap in a Retriever with scoring
        vector_retriever = VectorStoreRetrieverWithScore(
            vectorstore=vector_store,
            search_kwargs={"k": 3}
        )

        # 4. Prepare BM25 keyword retriever
        #    (tokenize docs for BM25Okapi under the hood)
        keyword_retriever = BM25Retriever.from_documents(
            langchain_docs,
            k1=1.5,
            b=0
        )

        # 5. Combine into an ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5],
        )
        logger.info("EnsembleRetriever (FAISS + BM25) is ready.")
        return self.ensemble_retriever


def load_ensemble_retriever() -> EnsembleRetriever:
    """
    Singleton pattern: initialize once, then reuse.
    """
    global ensemble_retriever_global
    if ensemble_retriever_global is None:
        logger.info("Initializing ensemble retriever...")
        manager = RetrieverManager()
        ensemble_retriever_global = manager.load_retriever()
    else:
        logger.info("Ensemble retriever already loaded; reusing.")
    return ensemble_retriever_global
