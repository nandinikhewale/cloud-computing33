"""
vector_store.py - Vector Database Module (RAG-9)
Manages local vector storage and indexing using ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "college_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "College documents semantic index for RAG"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Adds text chunks with precomputed embeddings and metadata into ChromaDB.
        """
        if not chunks:
            return 0

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": str(chunk.get("source", "unknown")),
                "page": int(chunk.get("page", 1)),
                "total_pages": int(chunk.get("total_pages", 1)),
                "start_char": int(chunk.get("start_char", 0)),
                "end_char": int(chunk.get("end_char", 0)),
            }
            for chunk in chunks
        ]

        # Chroma has a batch limit of 5461 elements
        batch_size = 500
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            b_ids = ids[i:i + batch_size]
            b_docs = documents[i:i + batch_size]
            b_embs = embeddings[i:i + batch_size]
            b_meta = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=b_ids,
                documents=b_docs,
                embeddings=b_embs,
                metadatas=b_meta
            )
            total_added += len(b_ids)

        return total_added

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB for the closest chunks to the query_embedding.
        Returns sorted list of matches with text, source, page, distance/similarity.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance or L2 distance conversion to approximate similarity
                # For cosine distance: similarity = 1 - distance
                similarity = max(0.0, 1.0 - float(dist)) if dist <= 1.0 else (1.0 / (1.0 + float(dist)))
                matches.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "total_pages": meta.get("total_pages", 1),
                    "distance": float(dist),
                    "similarity": round(similarity, 4)
                })

        return matches

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of the indexed vector database.
        """
        count = self.collection.count()
        sources = set()
        if count > 0:
            try:
                all_data = self.collection.get(include=["metadatas"])
                if all_data and "metadatas" in all_data:
                    for m in all_data["metadatas"]:
                        if m and "source" in m:
                            sources.add(m["source"])
            except Exception:
                pass

        return {
            "total_chunks": count,
            "unique_documents": list(sources),
            "storage_path": self.persist_directory
        }

    def clear(self):
        """
        Deletes all entries in the collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


if __name__ == "__main__":
    store = VectorStore()
    print("Vector Store initialized. Stats:", store.get_stats())
