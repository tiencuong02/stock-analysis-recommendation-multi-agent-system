import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)

# Namespace values stored as payload field — mirrors Pinecone namespace pattern
NAMESPACE_ADVISORY  = "internal-advisory"
NAMESPACE_KNOWLEDGE = "public-knowledge"
NAMESPACE_FAQ       = "faq-complaint"

# Payload field name used to segregate namespaces within the single collection
_NS_FIELD = "namespace"

# Similarity thresholds (cosine) — same calibration as before
SIMILARITY_THRESHOLD_ADVISORY  = 0.45
SIMILARITY_THRESHOLD_KNOWLEDGE = 0.40
SIMILARITY_THRESHOLD_DEFAULT   = 0.35

_CROSS_ENCODER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "BAAI/bge-reranker-v2-m3",
]


class VectorStoreService:
    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._store: Optional[QdrantVectorStore] = None
        self._cross_encoder: Optional[CrossEncoder] = None

        self._init_embeddings()
        self._init_qdrant()
        self._init_cross_encoder()

    # ─── Init ─────────────────────────────────────────────────────────────────

    def _init_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
            )
            logger.info(f"Embeddings: {settings.EMBEDDING_MODEL_NAME} initialized ({settings.EMBEDDING_DIMENSION} dims).")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = None

    def _init_qdrant(self):
        if self.embeddings is None:
            logger.warning("Embeddings not ready — skipping Qdrant init.")
            return
        try:
            self._client = QdrantClient(url=settings.QDRANT_URL, timeout=30)
            collection = settings.QDRANT_COLLECTION_NAME

            # Create collection only if it doesn't exist yet
            existing = {c.name for c in self._client.get_collections().collections}
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Qdrant: created collection '{collection}'.")
            else:
                logger.info(f"Qdrant: collection '{collection}' already exists.")

            self._store = QdrantVectorStore(
                client=self._client,
                collection_name=collection,
                embedding=self.embeddings,
            )
            logger.info(f"Qdrant initialized at {settings.QDRANT_URL}, collection='{collection}'.")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")
            self._client = None
            self._store = None

    def _init_cross_encoder(self):
        for model_name in _CROSS_ENCODER_MODELS:
            try:
                self._cross_encoder = CrossEncoder(model_name)
                logger.info(f"Cross-encoder initialized: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Cross-encoder '{model_name}' failed: {e}")
        logger.error("All cross-encoder models failed — reranking disabled.")
        self._cross_encoder = None

    # ─── Upsert ──────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks_with_metadata: List[Dict[str, Any]],
        namespace: str = NAMESPACE_ADVISORY,
    ) -> bool:
        if self._store is None:
            logger.error("Qdrant store not initialized.")
            return False

        texts = [c["text"] for c in chunks_with_metadata]
        # Inject namespace into every payload so we can filter by it later
        metas = [{**c["metadata"], _NS_FIELD: namespace} for c in chunks_with_metadata]

        try:
            self._store.add_texts(texts=texts, metadatas=metas)
            logger.info(f"Upserted {len(texts)} chunks (namespace='{namespace}').")
            return True
        except Exception as e:
            logger.error(f"Upsert failed (namespace='{namespace}'): {e}")
            return False

    # ─── Search ──────────────────────────────────────────────────────────────

    def search_similar_documents(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
        use_reranking: bool = True,
    ) -> List[Any]:
        """Hybrid Search: Dense (cosine) + BM25 sparse → RRF → optional cross-encoder rerank."""
        if self._store is None:
            return []

        target_namespaces = namespaces or [NAMESPACE_ADVISORY]
        fetch_k = max(k * 2, 10) if use_reranking else max(k + 5, 8)

        dense_docs_scored: List[Tuple[Any, float]] = []
        seen_keys: set = set()

        for ns in target_namespaces:
            qdrant_filter = self._build_filter(ns, filter_metadata)
            try:
                results = self._store.similarity_search_with_score(
                    query=query,
                    k=fetch_k,
                    filter=qdrant_filter,
                )
                for doc, score in results:
                    if len(doc.page_content.strip()) <= 10:
                        continue
                    key = (doc.page_content[:100], doc.metadata.get("page"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        dense_docs_scored.append((doc, float(score)))
            except Exception as e:
                logger.error(f"Dense search failed (namespace='{ns}'): {e}")

        if not dense_docs_scored:
            return []

        # Similarity threshold filter
        filtered = [(doc, sc) for doc, sc in dense_docs_scored if sc >= similarity_threshold]
        if not filtered:
            logger.info(
                f"All {len(dense_docs_scored)} docs below threshold {similarity_threshold}. "
                "No results returned."
            )
            return []

        # BM25 sparse re-scoring
        texts = [doc.page_content for doc, _ in filtered]
        bm25_scores = self._bm25_score(query, texts)

        # RRF fusion
        rrf_scores = self._rrf_fusion(
            dense_scores=[sc for _, sc in filtered],
            sparse_scores=bm25_scores,
        )

        candidates = sorted(
            zip([doc for doc, _ in filtered], rrf_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:fetch_k]

        # Cross-encoder rerank
        if use_reranking and self._cross_encoder is not None and len(candidates) > 1:
            docs_only = [doc for doc, _ in candidates]
            final = self._rerank(query, docs_only)[:k]
        else:
            final = [doc for doc, _ in candidates[:k]]

        # Attach similarity score for downstream consumers
        scores_map = {id(doc): sc for doc, sc in filtered}
        for doc in final:
            doc.metadata["_similarity_score"] = scores_map.get(id(doc), similarity_threshold)

        logger.info(
            f"VectorStore: {len(dense_docs_scored)} dense → "
            f"{len(filtered)} passed threshold → "
            f"{len(final)} after rerank."
        )
        return final

    def search_advisory(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        return self.search_similar_documents(
            query=query, k=k, filter_metadata=filter_metadata,
            namespaces=[NAMESPACE_ADVISORY],
            similarity_threshold=SIMILARITY_THRESHOLD_ADVISORY,
            use_reranking=True,
        )

    def search_knowledge(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        return self.search_similar_documents(
            query=query, k=k, filter_metadata=filter_metadata,
            namespaces=[NAMESPACE_KNOWLEDGE],
            similarity_threshold=SIMILARITY_THRESHOLD_KNOWLEDGE,
            use_reranking=False,
        )

    def search_faq(self, query: str, k: int = 3) -> List[Any]:
        return self.search_similar_documents(
            query=query, k=k,
            namespaces=[NAMESPACE_FAQ],
            similarity_threshold=0.72,
            use_reranking=False,
        )

    # ─── Delete ──────────────────────────────────────────────────────────────

    def delete_by_metadata(
        self,
        filter_metadata: Dict[str, Any],
        namespaces: Optional[List[str]] = None,
    ) -> bool:
        if self._client is None:
            logger.error("Qdrant client not initialized.")
            return False

        target = namespaces or [NAMESPACE_ADVISORY]
        success = True
        collection = settings.QDRANT_COLLECTION_NAME

        for ns in target:
            combined = {**filter_metadata, _NS_FIELD: ns}
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in combined.items()
                ]
            )
            try:
                self._client.delete(
                    collection_name=collection,
                    points_selector=FilterSelector(filter=qdrant_filter),
                )
                logger.info(f"Deleted vectors (namespace='{ns}', filter={filter_metadata}).")
            except Exception as e:
                logger.warning(f"Delete failed (namespace='{ns}'): {e}")
                success = False
        return success

    # ─── BM25 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _bm25_score(query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        tokenized_corpus = [t.lower().split() for t in texts]
        tokenized_query = query.lower().split()
        try:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query).tolist()
            max_s = max(scores) if max(scores) > 0 else 1.0
            return [s / max_s for s in scores]
        except Exception as e:
            logger.warning(f"BM25 scoring failed: {e}")
            return [0.0] * len(texts)

    # ─── RRF ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        dense_scores: List[float],
        sparse_scores: List[float],
        k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[float]:
        n = len(dense_scores)
        dense_ranks  = _rank_list(dense_scores)
        sparse_ranks = _rank_list(sparse_scores)
        return [
            dense_weight  * (1.0 / (k + dense_ranks[i]))
            + sparse_weight * (1.0 / (k + sparse_ranks[i]))
            for i in range(n)
        ]

    # ─── Rerank ──────────────────────────────────────────────────────────────

    def _rerank(self, query: str, docs: List[Any]) -> List[Any]:
        try:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self._cross_encoder.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return docs

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(namespace: str, extra: Optional[Dict[str, Any]] = None) -> Filter:
        """Build a Qdrant Filter combining namespace + optional extra payload conditions."""
        conditions = [FieldCondition(key=_NS_FIELD, match=MatchValue(value=namespace))]
        if extra:
            for k, v in extra.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        return Filter(must=conditions)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rank_list(scores: List[float]) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks
