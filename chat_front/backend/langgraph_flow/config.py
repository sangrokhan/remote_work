import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "http://localhost:19530")
DB_NAME = os.getenv("MILVUS_DB_NAME", "default")

BM25_search = os.getenv("BM25_SEARCH", "false").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "bge")
RERANKER_BGE_URL = os.getenv("RERANKER_BGE_URL", "")
RERANKER_QWEN3_URL = os.getenv("RERANKER_QWEN3_URL", "")
RERANKER_MAX_TOP_N = int(os.getenv("RERANKER_MAX_TOP_N", "10"))
RERANKER_THRESHOLD_RATIO = float(os.getenv("RERANKER_THRESHOLD_RATIO", "0.8"))

HYBRID_SEARCH_WEIGHTS = {
    "meta_sparse": float(os.getenv("WEIGHT_META_SPARSE", "0.4")),
    "contents_dense": float(os.getenv("WEIGHT_CONTENTS_DENSE", "0.3")),
    "contents_sparse": float(os.getenv("WEIGHT_CONTENTS_SPARSE", "0.2")),
    "meta_dense": float(os.getenv("WEIGHT_META_DENSE", "0.1")),
}

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bgem3")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "")
BGEM3_BASE_URL = os.getenv("BGEM3_BASE_URL", "")
BGEM3_API_KEY = os.getenv("BGEM3_API_KEY", "")
