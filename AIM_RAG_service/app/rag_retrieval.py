"""
RAG (Retrieval-Augmented Generation) vector store and retrieval operations
"""
import time

import numpy as np
from langchain_core.documents import Document
from app.mongo_client import get_mongo_collection, _to_python_types
from app.order_ask.checkpoint import checkpoint


class MongoRetriever:
    """Retriever that uses MongoDB vector store for similarity search"""

    def __init__(
            self,
            vectorstore,
            search_kwargs=None
    ):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {}

    def invoke(self, query: str):
        """Retrieve documents similar to the query"""
        k = self.search_kwargs.get("k", 15)
        fetch_k = self.search_kwargs.get("fetch_k", max(k, 15))

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )


class MongoVectorStore:
    """MongoDB-based vector store for semantic search with embeddings"""

    def __init__(
            self,
            collection,
            embeddings,
            namespace
    ):
        self.collection = collection
        self.embeddings = embeddings
        self.namespace = namespace

    def as_retriever(
            self,
            search_type="mmr",
            search_kwargs=None
    ):
        """Get a retriever from this vector store"""
        # Current implementation uses cosine similarity + reranking.
        # The search_type argument is accepted for API compatibility.
        _ = search_type

        return MongoRetriever(
            vectorstore=self,
            search_kwargs=search_kwargs
        )

    def add_documents(self, docs):
        """Add documents with embeddings to the vector store"""
        payload = []

        for doc in docs:
            metadata = _to_python_types(
                dict(doc.metadata)
            )

            chunk_embedding = metadata.pop(
                "chunk_embedding",
                None
            )

            if chunk_embedding is None:
                chunk_embedding = self.embeddings.embed_query(
                    doc.page_content[:4000]
                )

            chunk_embedding = [
                float(x)
                for x in chunk_embedding
            ]

            payload.append({
                "namespace": self.namespace,
                "page_content": doc.page_content,
                "embedding": chunk_embedding,
                "metadata": metadata
            })

        if payload:
            self.collection.insert_many(payload)

    def similarity_search(
            self,
            query,
            k=15,
            fetch_k=60
    ):
        """Search for documents similar to the query using cosine similarity.

        Two-pass on purpose: pass 1 pulls ONLY `_id` + `embedding` for every
        document in the namespace (needed to rank them) — not the full
        page_content/metadata, which for order/trip/invoice docs can be
        several KB EACH. Pass 2 then fetches page_content/metadata for just
        the k winners. Previously this pulled full page_content + metadata +
        embedding for every single document in the namespace on every query
        (regardless of k), and returned the full embedding vector in every
        result's metadata even though nothing downstream ever reads it back
        — both were pure overhead once the score is computed here.
        """
        t0 = time.perf_counter()

        query_embedding = np.array(
            self.embeddings.embed_query(query)
        )
        t_embed = time.perf_counter()

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        # Pass 1: rank — embedding only, not the full document.
        rank_docs = list(
            self.collection.find(
                {"namespace": self.namespace, "embedding": {"$exists": True, "$ne": None}},
                {"_id": 1, "embedding": 1}
            )
        )
        t_rank_fetch = time.perf_counter()

        if not rank_docs:
            checkpoint(
                "RAG_TIMING", "similarity_search (no candidates)",
                namespace=self.namespace,
                embed_ms=int((t_embed - t0) * 1000),
                rank_fetch_ms=int((t_rank_fetch - t_embed) * 1000),
            )
            return []

        embeddings_matrix = np.array([doc["embedding"] for doc in rank_docs])
        doc_norms = np.linalg.norm(embeddings_matrix, axis=1)
        dot_products = np.dot(embeddings_matrix, query_embedding)

        denoms = query_norm * doc_norms
        scores = np.zeros_like(dot_products)
        valid_denoms = denoms > 0
        scores[valid_denoms] = dot_products[valid_denoms] / denoms[valid_denoms]

        scored = sorted(zip(scores, rank_docs), key=lambda x: x[0], reverse=True)
        top = scored[:k]
        t_score = time.perf_counter()

        # Pass 2: fetch only the winners' page_content/metadata — no embedding.
        top_ids = [item["_id"] for _, item in top]
        detail_docs = {
            d["_id"]: d
            for d in self.collection.find(
                {"_id": {"$in": top_ids}},
                {"page_content": 1, "metadata": 1}
            )
        }
        t_detail_fetch = time.perf_counter()

        checkpoint(
            "RAG_TIMING", "similarity_search",
            namespace=self.namespace,
            candidates=len(rank_docs),
            returned=len(top),
            embed_ms=int((t_embed - t0) * 1000),
            rank_fetch_ms=int((t_rank_fetch - t_embed) * 1000),
            score_ms=int((t_score - t_rank_fetch) * 1000),
            detail_fetch_ms=int((t_detail_fetch - t_score) * 1000),
            total_ms=int((t_detail_fetch - t0) * 1000),
        )

        results = []
        for score, item in top:
            detail = detail_docs.get(item["_id"])
            if not detail:
                continue
            results.append(
                Document(
                    page_content=detail.get("page_content", ""),
                    metadata={
                        **detail.get("metadata", {}),
                        "similarity_score": float(score)
                    }
                )
            )
        return results


def get_vectorstore(
        embeddings,
        _unused_url,
        collection_name,
        _docs=None,
        mongo_collection_name=None,
        replace_namespace=True
):
    """Get or create a MongoDB vector store for a given collection"""
    try:
        collection = get_mongo_collection(mongo_collection_name)

        vectorstore = MongoVectorStore(
            collection=collection,
            embeddings=embeddings,
            namespace=collection_name
        )

        # --------------------------------
        # LOAD EXISTING NAMESPACE
        # --------------------------------

        if _docs is None:
            exists = collection.count_documents(
                {"namespace": collection_name},
                limit=1
            ) > 0

            return vectorstore if exists else None

        # --------------------------------
        # RECREATE NAMESPACE DATA
        # --------------------------------

        if replace_namespace:
            collection.delete_many({"namespace": collection_name})

        vectorstore.add_documents(_docs)

        return vectorstore

    except Exception as e:
        print(f"Vector store error: {str(e)}")
        return None
