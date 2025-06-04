import os
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models")
DATA_PATH  = os.getenv("DATA_PATH",  "/workspace/data_ingestion") 
# ─── diagnostics ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("CUDA detected")
    for i in range(torch.cuda.device_count()):
        print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not detected – abort")
    raise SystemExit

# ─── embeddings (GPU0) ───────────────────────────────────────────────────────
EMBEDDING_MODEL = SentenceTransformer(
    os.path.join(MODEL_PATH, "st_model"),
    device="cuda:0",
    cache_folder=MODEL_PATH
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.path.join(MODEL_PATH, "embed_model"),
    model_kwargs={"device": "cuda:0", "trust_remote_code": True},
    cache_folder=MODEL_PATH
)

# ─── generator LLM (GPU0, BF16, Flash-Attn-2) ───────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_PATH, "qwen_model"),
    trust_remote_code=True,
    cache_dir=MODEL_PATH
)

llm = AutoModelForCausalLM.from_pretrained(
    os.path.join(MODEL_PATH, "qwen_model"),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=MODEL_PATH
)

# ─── global placeholders ──────────────────────────────────────────────────────
ensemble_retriever_global = None
TAVILY_API_KEY            = "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"
