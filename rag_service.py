"""
rag_service.py - End-to-End RAG Service (RAG-11, RAG-13, RAG-14)
Coordinates Document Ingestion, Semantic Retrieval, LLM Response Generation,
and Source Citation Tracking.
"""

import os
from typing import List, Dict, Any, Optional
from document_loader import read_pdf, load_all_pdfs
from chunker import chunk_documents, create_chunks
from embeddings import generate_batch_embeddings, generate_embedding
from vector_store import VectorStore
from retriever import Retriever
from llm_service import build_rag_prompt, generate_llm_answer, is_ollama_available


class RAGService:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embed_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:3b"
    ):
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.vector_store = VectorStore(persist_directory=self.persist_dir)
        self.retriever = Retriever(vector_store=self.vector_store, embed_model=self.embed_model)

    def ingest_documents_from_directory(self, data_dir: str = "./data") -> Dict[str, Any]:
        """
        Loads all PDFs from data_dir, chunks them, generates embeddings,
        and saves them in ChromaDB.
        """
        pages = load_all_pdfs(data_dir)
        if not pages:
            return {"status": "empty", "chunks_added": 0, "message": f"No PDF documents found in {data_dir}"}

        chunks = chunk_documents(pages, chunk_size=1200, overlap=200)
        chunk_texts = [c["text"] for c in chunks]

        # Generate embeddings
        embeddings = generate_batch_embeddings(chunk_texts, model_name=self.embed_model)

        # Store in ChromaDB
        added = self.vector_store.add_chunks(chunks, embeddings)

        return {
            "status": "success",
            "pages_processed": len(pages),
            "chunks_added": added,
            "documents": list(set(p["source"] for p in pages))
        }

    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Loads a single uploaded PDF file into the vector store.
        """
        pages = read_pdf(file_path)
        if not pages:
            return {"status": "empty", "chunks_added": 0}

        chunks = chunk_documents(pages, chunk_size=1200, overlap=200)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_batch_embeddings(chunk_texts, model_name=self.embed_model)
        added = self.vector_store.add_chunks(chunks, embeddings)

        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "pages": len(pages),
            "chunks_added": added
        }

    def answer_question(self, user_question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline for a user question:
        1. Semantic retrieval of top-k chunks
        2. Context construction
        3. LLM generation
        4. Unknown answer verification
        5. Source extraction
        """
        user_question = user_question.strip()
        if not user_question:
            return {
                "question": "",
                "answer": "Please enter a valid question.",
                "sources": [],
                "retrieved_chunks": [],
                "is_safe_refusal": False
            }

        # Step 1: Semantic Retrieval
        retrieved_chunks = self.retriever.retrieve(user_question, top_k=top_k)

        # Step 2: Context formatting
        context, sources = self.retriever.format_context(retrieved_chunks)

        # Step 3: Prompt Construction
        prompt = build_rag_prompt(user_question, context)

        # Step 4: LLM Generation
        llm_response = generate_llm_answer(prompt, model_name=self.llm_model)
        raw_answer = llm_response.get("answer", "")

        # Step 5: Hallucination & Unknown-Answer check
        refusal_phrases = [
            "information not found in the available documents",
            "not found in the available documents",
            "not available in the provided context",
            "the provided text does not contain",
            "the context does not provide",
            "information is not available"
        ]
        is_safe_refusal = any(p in raw_answer.lower() for p in refusal_phrases)

        # Format final source citation
        if is_safe_refusal:
            final_sources = []
        else:
            final_sources = sources

        return {
            "question": user_question,
            "answer": raw_answer,
            "sources": final_sources,
            "retrieved_chunks": retrieved_chunks,
            "is_safe_refusal": is_safe_refusal,
            "engine": llm_response.get("engine", "Ollama"),
            "llm_status": llm_response.get("status", "unknown")
        }


if __name__ == "__main__":
    service = RAGService()
    print("RAG Service initialized successfully.")
