"""
embeddings.py - Vector Embedding Generation Module (RAG-8)
Generates vector representations for text chunks and queries.
Primary: Ollama 'nomic-embed-text' (Zero cost, runs locally)
Fallback: Local SentenceTransformer or deterministic semantic vectors if Ollama is offline.
"""

from typing import List, Union
import hashlib
import numpy as np
import time

# Global cache for fallback model and connection status
_FALLBACK_MODEL = None
_OLLAMA_STATUS_CACHE = {"online": None, "last_checked": 0}


def check_ollama_online(timeout_sec: float = 0.3) -> bool:
    """
    Fast-fail check if Ollama daemon is listening on localhost:11434.
    Caches result for 3 seconds to avoid socket overhead.
    """
    now = time.time()
    if _OLLAMA_STATUS_CACHE["online"] is not None and (now - _OLLAMA_STATUS_CACHE["last_checked"]) < 3.0:
        return _OLLAMA_STATUS_CACHE["online"]

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_sec)
    try:
        s.connect(("127.0.0.1", 11434))
        s.close()
        _OLLAMA_STATUS_CACHE["online"] = True
    except Exception:
        _OLLAMA_STATUS_CACHE["online"] = False

    _OLLAMA_STATUS_CACHE["last_checked"] = now
    return _OLLAMA_STATUS_CACHE["online"]


def get_ollama_embedding(text: str, model_name: str = "nomic-embed-text") -> List[float]:
    """
    Attempts to generate an embedding using a local Ollama instance.
    Handles both modern ollama.embed() and older ollama.embeddings() APIs.
    """
    if not check_ollama_online():
        raise RuntimeError("Ollama daemon is not reachable on localhost:11434")

    import ollama

    # Strip and format text
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        clean_text = "empty"

    try:
        # Modern Ollama Python API: ollama.embed
        if hasattr(ollama, "embed"):
            res = ollama.embed(model=model_name, input=clean_text)
            if "embeddings" in res and res["embeddings"]:
                return res["embeddings"][0]
        # Older Ollama Python API: ollama.embeddings
        res = ollama.embeddings(model=model_name, prompt=clean_text)
        return res["embedding"]
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed for model '{model_name}': {e}")


def get_fallback_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Fallback embedding generator:
    1. Tries local sentence_transformers ('all-MiniLM-L6-v2') if installed.
    2. Otherwise uses deterministic normalized hash projection (ensures 0-dependency offline fallback).
    """
    global _FALLBACK_MODEL

    try:
        if _FALLBACK_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _FALLBACK_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _FALLBACK_MODEL.encode(text, convert_to_numpy=True)
        return emb.tolist()
    except Exception:
        # Robust lightweight deterministic semantic hash vectorizer
        # Generates a consistent unit vector of length `dim` for any input
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        if not words:
            return vec.tolist()

        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            pos = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[pos] += sign * (1.0 / (1.0 + (i * 0.05)))

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def generate_embedding(text: str, model_name: str = "nomic-embed-text", use_fallback_if_offline: bool = True) -> List[float]:
    """
    Primary interface for generating single embeddings.
    Tries Ollama first. Falls back if Ollama daemon is not running.
    """
    try:
        return get_ollama_embedding(text, model_name=model_name)
    except Exception as e:
        if use_fallback_if_offline:
            return get_fallback_embedding(text)
        raise e


def generate_batch_embeddings(texts: List[str], model_name: str = "nomic-embed-text") -> List[List[float]]:
    """
    Generates embeddings for a batch of strings.
    """
    return [generate_embedding(t, model_name=model_name) for t in texts]


if __name__ == "__main__":
    test_str = "minimum attendance requirement"
    print("Testing embedding generator...")
    emb = generate_embedding(test_str)
    print(f"Generated embedding of length {len(emb)} for: '{test_str}'")
