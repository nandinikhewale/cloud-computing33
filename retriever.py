"""
retriever.py - Semantic Retrieval Module (RAG-10)
Translates user queries into embeddings and retrieves top-k matching document chunks.
"""

from typing import List, Dict, Any, Tuple
from embeddings import generate_embedding
from vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore = None, embed_model: str = "nomic-embed-text"):
        self.vector_store = vector_store or VectorStore()
        self.embed_model = embed_model

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.15) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks for the given question.
        Returns a list of dicts with text, source, page, similarity score.
        """
        if not query or not query.strip():
            return []

        # Step 1: Question -> Vector Embedding
        query_vector = generate_embedding(query, model_name=self.embed_model)

        # Step 2: Semantic vector search in ChromaDB
        candidates = self.vector_store.query_similar(query_vector, top_k=top_k)

        # Step 3: Optional score threshold filtering
        filtered = [c for c in candidates if c.get("similarity", 1.0) >= min_similarity]

        return filtered if filtered else candidates

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Builds a clean structured context string for LLM injection,
        and returns the list of unique cited sources.
        """
        if not retrieved_chunks:
            return "", []

        context_blocks = []
        sources = set()

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page", 1)
            sources.add(f"{source} (Page {page})")

            context_blocks.append(
                f"[Document {idx}: {source} | Page {page}]\n{chunk.get('text', '').strip()}"
            )

        full_context = "\n\n".join(context_blocks)
        return full_context, list(sources)


if __name__ == "__main__":
    retriever = Retriever()
    chunks = retriever.retrieve("What is the attendance requirement?", top_k=2)
    ctx, srcs = retriever.format_context(chunks)
    print("Retrieved chunks:", len(chunks))
    print("Sources:", srcs)
