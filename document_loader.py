"""
document_loader.py - PDF Text Extraction Module (RAG-6)
Accepts PDF documents and extracts text from readable pages while preserving metadata.
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader


def read_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a single PDF file page by page.
    Returns a list of dicts: [{"text": str, "source": str, "page": int, "total_pages": int}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")

    filename = os.path.basename(file_path)
    extracted_pages = []

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()
            # Retain page even if short, but skip completely empty/whitespace-only pages
            if cleaned_text:
                extracted_pages.append({
                    "text": cleaned_text,
                    "source": filename,
                    "file_path": file_path,
                    "page": page_idx,
                    "total_pages": total_pages
                })
    except Exception as e:
        print(f"Error reading PDF {filename}: {e}")

    return extracted_pages


def load_all_pdfs(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scans a directory for all .pdf files and extracts text from all readable pages.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return []

    all_pages = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    for file_name in pdf_files:
        full_path = os.path.join(directory_path, file_name)
        pages = read_pdf(full_path)
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    import sys
    test_dir = "./data"
    print(f"Scanning {test_dir}...")
    pages = load_all_pdfs(test_dir)
    print(f"Extracted {len(pages)} pages across PDF documents.")
