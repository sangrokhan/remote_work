import torch
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
import logging

import langgraph_flow.config as config
from langgraph_flow.core.factory import get_embedding_provider

logger = logging.getLogger(__name__)

# --- Configuration ---
# Get embedding provider from registry (proper resource management)
embedding_provider = get_embedding_provider()

description = {
    "type": "function",
    "function": {
        "name": "retriever",
        "description": """Retrieves algorithmic details and parameter descriptions from the features' documents.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to perform in *English*. This should be semantically close to your target documents. Use an affirmative statement rather than a question."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents from the highest ranking ones to be considered. If the user wants more detailed information, increase 'top_k' to 15 or 20; otherwise, decrease 'top_k' to 5.",
                    "default": 10
                }
            },
            "required": ["query"],
        },
    },
}


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def format_dense_vector(dense_tensor):
    """
    Milvus가 인식할 수 있도록 dense vector 형식 변환
    모든 값을 Python native float로 변환하여 NumPy float32, torch float32 등의 타입 문제 해결
    None이나 변환 불가능한 값은 필터링
    """
    formatted = []

    # dense_tensor는 tensor들의 리스트여야 함
    for vector in dense_tensor:
        # torch tensor나 numpy array를 list로 변환
        if hasattr(vector, 'tolist'):
            vector_list = vector.tolist()
        elif isinstance(vector, list):
            vector_list = vector
        else:
            vector_list = list(vector)

        # 2D tensor인 경우 flatten (예: [[1,2,3]] -> [1,2,3])
        if len(vector_list) > 0 and isinstance(vector_list[0], list):
            # 첫 번째 요소가 리스트면 2D tensor로 간주하고 flatten
            vector_list = [item for sublist in vector_list for item in sublist]

        clean_vector = []

        for val in vector_list:
            try:
                if val is None:
                    logger.debug(f"Skipping None value in dense vector")
                    continue
                float_val = float(val)
                if float_val != float_val:  # NaN 체크
                    logger.debug(f"Skipping NaN value in dense vector")
                    continue
                if float_val == float('inf') or float_val == float('-inf'):
                    logger.debug(f"Skipping inf value in dense vector")
                    continue
                clean_vector.append(float_val)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid value in dense vector: {val} - Error: {e}")
                continue

        formatted.append(clean_vector)

    return formatted


def format_sparse_vector(sparse_dicts):
    """
    Milvus가 인식할 수 있도록 sparse vector 형식 변환
    인덱스는 int, 값은 float로 변환하여 NumPy float32 등의 타입 문제 해결
    None이나 변환 불가능한 값은 필터링
    """
    formatted = []
    for sparse in sparse_dicts:
        # 인덱스는 int, 값은 float로 변환
        # None이나 변환 불가능한 값은 필터링
        new_dict = {}
        for k, v in sparse.items():
            try:
                if v is None:
                    logger.debug(f"Skipping None value at key {k}")
                    continue
                float_val = float(v)
                if float_val != float_val:  # NaN 체크
                    logger.debug(f"Skipping NaN value at key {k}")
                    continue
                if float_val == float('inf') or float_val == float('-inf'):
                    logger.debug(f"Skipping inf value at key {k}")
                    continue
                new_dict[int(k)] = float_val
                logger.debug(f"Added entry: {int(k)} -> {float_val}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid entry {k}: {v} - Error: {e}")
                continue
        formatted.append(new_dict)
    return formatted


# ==================== Search Strategy Pattern ====================

class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    Different embedding providers can implement their own search strategies.
    """

    @abstractmethod
    def perform_search(
            self,
            milvus_client: MilvusClient,
            collection_name: str,
            query: str,
            query_dense: torch.Tensor,
            query_sparse: Optional[list],
            search_params: Dict[str, Any],
            top_k: int
    ) -> list:
        """
        Perform search on a collection.
        
        Args:
            milvus_client: Milvus client
            collection_name: Collection name to search
            query_dense: Dense query vector
            query_sparse: Sparse query vector (optional)
            search_params: Search parameters
            top_k: Number of results to return
        
        Returns:
            Search results
        """
        pass

    @abstractmethod
    def get_search_type(self) -> str:
        """
        Get the search type name for logging.
        
        Returns:
            Search type string
        """
        pass

    def _weighted_rrf_fusion(self, search_results: list, top_k: int) -> list:
        """
        Default RRF fusion implementation (no-op for dense-only strategies).
        Subclasses can override for custom fusion logic.
        
        Args:
            search_results: List of search results
            top_k: Number of results to return
            
        Returns:
            Results (as-is for default implementation)
        """
        # Default: just return the results as-is (for dense-only)
        # This is a no-op for strategies that don't need RRF fusion
        if search_results and isinstance(search_results, list):
            if isinstance(search_results[0], str):
                # Already text results, return as-is
                return search_results[:top_k]
        return search_results[:top_k] if search_results else []


class BGE3SearchStrategy(SearchStrategy):
    """
    Search strategy for BGE3 embedding provider.
    Uses 4-way hybrid search with separate fields for meta and contents.
    Implements Weighted RRF algorithm for score fusion.
    """

    def _weighted_rrf_fusion(self, search_results: list, top_k: int) -> list:
        """
        Apply Weighted RRF (Reciprocal Rank Fusion) algorithm to fuse 4-way search results.
        
        Weight configuration from config.HYBRID_SEARCH_WEIGHTS:
        - Meta Sparse (MS): 0.4 (타겟 버전 및 파라미터명 정확도 최우선)
        - Contents Dense (CD): 0.3 (릴리즈 히스토리 및 의미적 상호작용 확보)
        - Contents Sparse (CS): 0.2 (전문 용어 및 약어 매칭 보정)
        - Meta Dense (MD): 0.1 (Feature Category 기반 개념적 연결고리)
        
        Args:
            search_results: List of search results from different vector fields
            top_k: Number of final results to return
            
        Returns:
            Fused results sorted by weighted RRF scores
        """
        from collections import defaultdict

        # Get weights from config
        weights = config.HYBRID_SEARCH_WEIGHTS
        k = 60  # RRF constant

        # Dictionary to store document scores
        doc_scores = defaultdict(float)
        doc_entities = {}  # Store document entities for final output

        # Process each search result (each corresponds to a different vector field)
        # Order: [meta_sparse, contents_dense, contents_sparse, meta_dense]
        field_weights = [
            weights["meta_sparse"],  # 0.4
            weights["contents_dense"],  # 0.3
            weights["contents_sparse"],  # 0.2
            weights["meta_dense"]  # 0.1
        ]

        for i, result in enumerate(search_results):
            if not result:
                continue

            # Apply weight based on field type
            field_weight = field_weights[i] if i < len(field_weights) else 1.0
            for rank, item in enumerate(result):
                doc_id = item.get('id', f"doc_{i}_{rank}")
                entity = item.get('entity', {})

                # Calculate RRF score: 1 / (rank + k)
                rrf_score = 1.0 / (rank + 1 + k)

                # Apply field weight
                weighted_score = rrf_score * field_weight

                doc_scores[doc_id] += weighted_score
                doc_entities[doc_id] = entity

        # Sort by final scores and return top_k results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results = []

        for doc_id, score in sorted_docs[:top_k]:
            fused_results.append({
                'id': doc_id,
                'distance': score,  # Use fused score as distance
                'entity': doc_entities[doc_id]
            })

        return [fused_results]

    def perform_search(
            self,
            milvus_client: MilvusClient,
            collection_name: str,
            query: str,
            query_dense: torch.Tensor,
            query_sparse: Optional[list],
            search_params: Dict[str, Any],
            top_k: int
    ) -> list:
        """Perform 4-way hybrid search with meta and contents vectors using all available vector fields."""

        # Store individual search results for weighted RRF fusion
        reqs = []

        formatted_sparse = format_sparse_vector(query_sparse)
        formatted_dense = format_dense_vector([query_dense])

        # Add all available vector fields to the search request
        # Order: sparse_vector_meta, sparse_vector_contents, dense_vector_7, dense_vector_3

        # BM25 전체 텍스트 검색
        if "sparse_vector_bm25" in search_params:
            vector_field = "sparse_vector_bm25"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[query],
                anns_field=vector_field,
                limit=top_k,
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])
        # Sparse Vector Meta+Contents search
        if "sparse_vector_bge" in search_params and formatted_sparse:
            vector_field = "sparse_vector_bge"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[formatted_sparse[0]],
                anns_field=vector_field,
                limit=top_k,
                search_params=search_params[vector_field],
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])  # Sparse Vector Meta search
        if "sparse_vector_meta" in search_params and formatted_sparse:
            vector_field = "sparse_vector_meta"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[formatted_sparse[0]],
                anns_field=vector_field,
                limit=top_k,
                search_params=search_params[vector_field],
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])
        # Sparse Vector Contents search
        if "sparse_vector_contents" in search_params and formatted_sparse:
            vector_field = "sparse_vector_contents"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[formatted_sparse[0]],
                anns_field=vector_field,
                limit=top_k,
                search_params=search_params[vector_field],
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])

        # Dense Vector 7 search (Meta Dense)
        if "dense_vector_7" in search_params:
            vector_field = "dense_vector_7"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=formatted_dense,
                anns_field=vector_field,
                limit=top_k * 2,
                search_params=search_params[vector_field],
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])
        # Dense Vector 3 search (Contents Dense)
        if "dense_vector_3" in search_params:
            vector_field = "dense_vector_3"
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=formatted_dense,
                anns_field=vector_field,
                limit=top_k,
                search_params=search_params[vector_field],
                output_fields=["text", "feature_id", "feature_name"]
            )
            reqs += (search_res[0])
        return [reqs]

    def get_search_type(self) -> str:
        """Return search type for logging."""
        return "Hybrid (RRF)"


class DenseOnlySearchStrategy(SearchStrategy):
    """
    Search strategy for dense-only embedding providers (e.g., OpenAI).
    Uses only dense vectors for search.
    """

    def perform_search(
            self,
            milvus_client: MilvusClient,
            collection_name: str,
            query_dense: torch.Tensor,
            query_sparse: Optional[list],
            search_params: Dict[str, Any],
            top_k: int
    ) -> list:
        """Perform dense-only search."""
        vector_field = list(search_params.keys())[0]

        return milvus_client.search(
            collection_name=collection_name,
            data=query_dense.tolist(),
            anns_field=vector_field,
            limit=top_k,
            search_params=search_params[vector_field],
            output_fields=["text", "feature_id", "feature_name"]
        )

    def get_search_type(self) -> str:
        """Return search type for logging."""
        return "Dense-only"


def get_search_strategy(embedding_provider) -> SearchStrategy:
    """
    Factory function to get the appropriate search strategy based on embedding provider.
    
    Args:
        embedding_provider: Embedding provider instance
    
    Returns:
        Appropriate SearchStrategy instance
    """
    collection_suffix = embedding_provider.get_collection_suffix()

    if collection_suffix == "_bge3":
        return BGE3SearchStrategy()
    else:
        return DenseOnlySearchStrategy()


# ==================== Reranking Functions ====================

def _apply_reranking(
        query: str,
        documents: list[dict],
        top_n: int = 5,
        reranker_provider: str = "bge",
        score_threshold_ratio: float = None
) -> list[dict]:
    """
    Apply reranking using the specified reranker service.

    Args:
        documents: list[dict] with keys {text, feature_id, feature_name}
    Returns:
        list[dict]: reranked docs preserving metadata
    """
    if not documents:
        return []

    text_to_meta = {d.get('text', ''): d for d in documents if isinstance(d, dict)}
    doc_texts = [d.get('text', '') for d in documents if isinstance(d, dict)]

    # Select reranker endpoint based on provider
    if reranker_provider == "bge":
        rerank_url = config.RERANKER_BGE_URL
        logger.info(f"Using BGE Reranker: {rerank_url}")
    elif reranker_provider == "qwen3":
        rerank_url = config.RERANKER_QWEN3_URL
        logger.info(f"Using Qwen3 Reranker: {rerank_url}")
    else:
        logger.warning(f"Unknown reranker provider: {reranker_provider}, defaulting to BGE")
        rerank_url = config.RERANKER_BGE_URL

    # Prepare request payload (reranker service expects plain text strings)
    payload = {
        "query": query,
        "documents": doc_texts,
        "top_n": min(top_n, len(doc_texts))
    }

    try:
        # Call reranker service
        response = requests.post(
            f"{rerank_url}rerank",
            json=payload,
            timeout=30,
            verify=False
        )
        response.raise_for_status()

        result = response.json()
        # Extract reranked documents with score-based filtering
        if "results" in result:
            # top score 기반 threshold 계산
            top_score = result['results'][0]['score']
            threshold = score_threshold_ratio or getattr(config, 'RERANKER_THRESHOLD_RATIO',
                                                         0.8)

            logger.debug(f"[RERANK] Reranking filter details:")
            logger.debug(f"[RERANK]   - Top score: {top_score:.4f}")
            logger.debug(f"[RERANK]   - Calculated threshold: {threshold:.4f}")
            logger.debug(f"[RERANK]   - Total input documents: {len(result['results'])}")

            # threshold 이상의 문서만 필터링, metadata 재첨부
            filtered_results = [r for r in result['results'] if r['score'] >= threshold]
            reranked_docs = []
            for r in filtered_results:
                text = r.get('text', '')
                meta = text_to_meta.get(text, {})
                reranked_docs.append({
                    'text': text,
                    'feature_id': meta.get('feature_id', ''),
                    'feature_name': meta.get('feature_name', ''),
                })

            # 필터링된 문서들의 score 정보 로그
            if filtered_results:
                scores = [r['score'] for r in filtered_results]
                logger.debug(
                    f"[RERANK]   - Filtered documents scores: {[f'{score:.4f}' for score in scores]}")
                logger.debug(f"[RERANK]   - Score range: {min(scores):.4f} ~ {max(scores):.4f}")

            logger.info(
                f"[RERANK] Top score: {top_score:.4f}, Threshold: {threshold:.4f}")
            logger.info(
                f"[RERANK] Successfully reranked {len(documents)} docs → {len(reranked_docs)} docs (filtered)")

            # 필터링된 문서 수에 따른 상세 로그
            if len(filtered_results) == 0:
                logger.warning(
                    f"[RERANK] All documents filtered out! Consider lowering threshold ratio.")
            elif len(filtered_results) == len(result['results']):
                logger.debug(f"[RERANK] All documents passed threshold filter")
            else:
                logger.debug(
                    f"[RERANK] {len(filtered_results)}/{len(result['results'])} documents passed threshold filter ({len(filtered_results) / len(result['results']) * 100:.1f}%)")

            return reranked_docs
        else:
            logger.warning("[RERANK] Invalid response format, returning original documents")
            return documents[:top_n]

    except requests.exceptions.Timeout:
        logger.warning("[RERANK] Reranker request timed out, returning original documents")
        return documents[:top_n]
    except requests.exceptions.RequestException as e:
        logger.warning(f"[RERANK] Reranker request failed: {e}, returning original documents")
        return documents[:top_n]
    except Exception as e:
        logger.error(f"[RERANK] Unexpected error during reranking: {e}")
        return documents[:top_n]


# ==================== Helper Functions ====================

def _filter_and_log_results(
        search_res: list,
) -> list:
    """
    검색 결과 필터링. metadata(feature_id, feature_name) 보존.

    Returns:
        list[dict]: [{text, feature_id, feature_name}], 동일 text dedup
    """
    seen_texts = set()
    output = []
    for o in search_res[0]:
        entity = o.get('entity', {}) if isinstance(o, dict) else {}
        text = entity.get('text', '')
        if text in seen_texts:
            continue
        seen_texts.add(text)
        output.append({
            'text': text,
            'feature_id': entity.get('feature_id', ''),
            'feature_name': entity.get('feature_name', ''),
        })
    return output


# ==================== Main Retriever Tool ====================

def RetrieverTool(query: str, top_k=10,
                  score_threshold=0.4,
                  ) -> list[dict]:
    """
    Retrieves relevant information from CSV files using RAG (Retrieval-Augmented Generation).
    The function processes each feature, finds the corresponding CSV file,
    extracts and splits documents, and then performs a search.
    Args:
        query: The query to search for.
        feature: A list of feature names to search within.
    Returns:
        A list of dictionaries, where each dictionary contains the results
        for a specific feature, including the feature name and retrieved documents.
        Returns a list with an error dictionary if an exception occurs.
    """
    logger.info(f"retriever argument: {query}, top_k: {top_k}")
    outputs = []

    # task = 'Given a query, retrieve relevant chunks that answer the query'
    task = ''
    origin_query = query
    queries = [
        get_detailed_instruct(task, query)
    ]

    # Use embedding provider to encode queries
    query_embeddings = embedding_provider.encode_queries(queries)
    if embedding_provider.get_collection_suffix() == "_bge3":
        query_dense = torch.tensor(query_embeddings["dense"])
        query_sparse = query_embeddings["sparse"]
    else:
        query_dense = torch.tensor(query_embeddings["vector"])
        query_sparse = None

    ######################################
    # Find technical details using only FD because required features are too many
    milvus_client = MilvusClient(config.MILVUS_HOST, db_name=config.DB_NAME)

    # Get search parameters from embedding provider
    search_params = embedding_provider.get_search_params(bm25=config.BM25_search)

    # Get the appropriate search strategy based on embedding provider
    search_strategy = get_search_strategy(embedding_provider)

    # Determine effective top_k for RRF search (fetch more if reranker will be used)
    # Limit to reasonable maximum to avoid performance issues
    rrf_top_k = min(top_k * 2, 20) if config.USE_RERANKER else top_k

    # Determine reranker top_n (dynamically adjusted based on user request)
    # Use user's top_k request, but cap at MAX_TOP_N for performance
    reranker_top_n = min(top_k, config.RERANKER_MAX_TOP_N) if config.USE_RERANKER else 0

    logger.info(f"[RETRIEVER] top_k={top_k}, rrf_top_k={rrf_top_k}, "
                f"use_reranker={config.USE_RERANKER}, reranker_top_n={reranker_top_n}")

    collection_name = 'feature_descriptions' + embedding_provider.get_collection_suffix()

    search_res = search_strategy.perform_search(
        milvus_client=milvus_client,
        collection_name=collection_name,
        query=query,
        query_dense=query_dense,
        query_sparse=query_sparse,
        search_params=search_params,
        top_k=rrf_top_k
    )
    # Apply LLM reranking if enabled, else RRF reranker is applied
    if config.USE_RERANKER and reranker_top_n > 0:
        logger.info(
            f"[RERANK] Applying reranking to full feature description results, top_n={reranker_top_n}")
        output = _filter_and_log_results(
            search_res=search_res,
        )
        output = _apply_reranking(
            query=origin_query,
            documents=output,
            top_n=reranker_top_n,
            reranker_provider=config.RERANKER_PROVIDER,
            score_threshold_ratio=config.RERANKER_THRESHOLD_RATIO
        )
        outputs += output
    else:
        logger.info(
            f"[W/O RERANK] DO NOT applying reranking, top_n={reranker_top_n}")
        output = search_strategy._weighted_rrf_fusion(search_results=search_res, top_k=top_k)
        output = _filter_and_log_results(
            search_res=output,
        )

        outputs.append(output)
    return outputs
