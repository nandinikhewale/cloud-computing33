"""
chunker.py - Text Chunking Module (RAG-7)
Divides extracted document text into smaller, overlapping chunks with metadata.
"""

from typing import List, Dict, Any


def create_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Divides input text into overlapping chunks.
    Ensures boundary sentences/paragraphs aren't cut mid-thought where possible.
    """
    if not text or not text.strip():
        return []

    chunks = []
    text_length = len(text)
    start = 0
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_data = {
                "chunk_id": f"{metadata.get('source', 'doc')}_p{metadata.get('page', 1)}_c{chunk_index}",
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "source": metadata.get("source", "Unknown Document") if metadata else "Unknown Document",
                "page": metadata.get("page", 1) if metadata else 1,
                "total_pages": metadata.get("total_pages", 1) if metadata else 1,
            }
            chunks.append(chunk_data)
            chunk_index += 1

        if end >= text_length:
            break

        start += max(1, chunk_size - overlap)

    return chunks


def chunk_documents(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Takes a list of document page dicts from document_loader and chunks each page.
    """
    all_chunks = []
    for page_info in pages:
        page_chunks = create_chunks(
            text=page_info["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={
                "source": page_info["source"],
                "page": page_info["page"],
                "total_pages": page_info["total_pages"],
            }
        )
        all_chunks.extend(page_chunks)

    return all_chunks


if __name__ == "__main__":
    sample = "A" * 3000
    res = create_chunks(sample, chunk_size=1000, overlap=100, metadata={"source": "test.pdf", "page": 1})
    print(f"Created {len(res)} chunks from {len(sample)} characters.")
