"""
RAG (Retrieval-Augmented Generation) vector store and retrieval operations
"""
import numpy as np
from langchain_core.documents import Document
from app.mongo_client import get_mongo_collection, _to_python_types


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
        """Search for documents similar to the query using cosine similarity"""
        query_embedding = np.array(
            self.embeddings.embed_query(query)
        )

        if np.linalg.norm(query_embedding) == 0:
            return []

        docs = list(
            self.collection.find(
                {"namespace": self.namespace},
                {
                    "_id": 0,
                    "page_content": 1,
                    "metadata": 1,
                    "embedding": 1
                }
            )
        )

        if not docs:
            return []

        # Vectorized similarity search optimization:
        # Instead of iterating sequentially in a slow Python loop, we construct a 2D matrix
        # and compute all cosine similarities in a single optimized NumPy linear algebra call.
        # This speeds up retrieval matching tremendously as documents increase.
        valid_docs = [doc for doc in docs if doc.get("embedding") is not None]
        if not valid_docs:
            return []

        embeddings_matrix = np.array([doc["embedding"] for doc in valid_docs])

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        doc_norms = np.linalg.norm(embeddings_matrix, axis=1)

        dot_products = np.dot(embeddings_matrix, query_embedding)

        denoms = query_norm * doc_norms
        scores = np.zeros_like(dot_products)
        valid_denoms = denoms > 0
        scores[valid_denoms] = dot_products[valid_denoms] / denoms[valid_denoms]

        scored = list(zip(scores, valid_docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        candidate_limit = max(k, fetch_k)
        top = scored[:candidate_limit]

        return [
            Document(
                page_content=item["page_content"],
                metadata={
                    **item.get("metadata", {}),
                    "embedding": item.get("embedding"),
                    "similarity_score": float(score)
                }
            )
            for score, item in top[:k]
        ]


def get_vectorstore(
        embeddings,
        _unused_url,
        collection_name,
        _docs=None
):
    """Get or create a MongoDB vector store for a given collection"""
    try:
        collection = get_mongo_collection()

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

        collection.delete_many({"namespace": collection_name})

        vectorstore.add_documents(_docs)

        return vectorstore

    except Exception as e:
        print(f"Vector store error: {str(e)}")
        return None
