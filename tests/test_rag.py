"""
test_rag.py - Unit and Integration Tests for RAG Components
Covers RAG-6, RAG-7, RAG-8, RAG-9, RAG-10, RAG-13, RAG-14.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_loader import read_pdf, load_all_pdfs
from chunker import create_chunks, chunk_documents
from embeddings import generate_embedding, get_fallback_embedding
from vector_store import VectorStore
from retriever import Retriever
from llm_service import build_rag_prompt, _generate_extractive_fallback


class TestRAGComponents(unittest.TestCase):

    def test_chunking(self):
        text = "This is sentence one. " * 100
        chunks = create_chunks(text, chunk_size=300, overlap=50, metadata={"source": "test.pdf", "page": 1})
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "test.pdf")
        self.assertIn("chunk_id", chunks[0])

    def test_fallback_embeddings(self):
        emb1 = get_fallback_embedding("attendance requirement")
        emb2 = get_fallback_embedding("attendance requirement")
        emb3 = get_fallback_embedding("placement criteria")
        self.assertEqual(len(emb1), 384)
        # Deterministic
        self.assertEqual(emb1, emb2)

    def test_vector_store_and_retrieval(self):
        v_store = VectorStore(persist_directory="./chroma_db_test", collection_name="test_col")
        v_store.clear()

        sample_chunks = [
            {"chunk_id": "c1", "text": "Students must maintain 75% attendance to write exams.", "source": "regulations.pdf", "page": 1},
            {"chunk_id": "c2", "text": "Library timings are 8am to 8pm.", "source": "handbook.pdf", "page": 2}
        ]
        embs = [get_fallback_embedding(c["text"]) for c in sample_chunks]
        added = v_store.add_chunks(sample_chunks, embs)
        self.assertEqual(added, 2)

        # Query
        q_emb = get_fallback_embedding("What is the attendance required?")
        matches = v_store.query_similar(q_emb, top_k=1)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "regulations.pdf")

        # Clean up
        v_store.clear()

    def test_hallucination_refusal(self):
        prompt = build_rag_prompt(
            user_question="What is the hostel mess fee?",
            retrieved_context="Academic regulations state that degree requires 160 credits."
        )
        res = _generate_extractive_fallback(prompt)
        self.assertIn("Information not found in the available documents.", res["answer"])


if __name__ == "__main__":
    unittest.main()
