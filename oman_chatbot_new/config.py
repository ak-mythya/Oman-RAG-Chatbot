# config.py
import nest_asyncio
nest_asyncio.apply()

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="omarelshehy/arabic-english-sts-matryoshka-v2.0"
)

DB_FAISS_PATH = "./vector_index/"
DATA_PATH = "./data_ingestion/data/"

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Groq LLM
groq_api_key = "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"

llama_llm = ChatGroq(temperature=0, model_name="Llama3-8b-8192", groq_api_key=groq_api_key)

# Ensemble Retriever global placeholder
ensemble_retriever_global = None

# Tavily
TAVILY_API_KEY = "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"
