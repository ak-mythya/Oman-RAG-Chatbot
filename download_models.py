# download_models.py

from huggingface_hub import snapshot_download

COMMON_IGNORE_PATTERNS = [
    "*.onnx",
    "*.pb",
    "*.h5",
    "*.tflite",
    "*.xet",
    "onnx/*",
    "openvino/*",
    ".git/*",
    ".gitattributes",
    ".cache/*",
    "*.md",
    "*.pdf",
    "*.txt",
    "*.tar*",
    "*.zip",
    "README*"
]

print("Downloading Qwen model...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/workspace/models/qwen_model",
    local_dir_use_symlinks=False,
    ignore_patterns=COMMON_IGNORE_PATTERNS
)

print("Downloading Arabic Triplet Matryoshka embed model...")
snapshot_download(
    repo_id="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    local_dir="/workspace/models/embed_model",
    local_dir_use_symlinks=False,
    ignore_patterns=COMMON_IGNORE_PATTERNS
)

print("Downloading Sentence Transformer model...")
snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir="/workspace/models/st_model",
    local_dir_use_symlinks=False,
    ignore_patterns=COMMON_IGNORE_PATTERNS
)

print("All models downloaded.")
