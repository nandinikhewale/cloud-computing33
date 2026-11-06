"""
llm_service.py - LLM Interface Module (RAG-5, RAG-11, RAG-12, RAG-14)
Interfaces with local Ollama runtime ('llama3.2:3b') and generates answers
strictly grounded in the retrieved context.
"""

from typing import Dict, Any, Optional
import os
import re
import json
import urllib.request
from embeddings import check_ollama_online


# Strict system prompt ensuring hallucination prevention & adherence to context
RAG_SYSTEM_PROMPT = """You are a college document information assistant.
Answer the user's question using ONLY the retrieved college document context provided below.
Rules:
1. Do not use any external knowledge or assumptions.
2. If the answer cannot be determined directly from the context, you MUST reply EXACTLY with:
"Information not found in the available documents."
3. Give a concise, factual, and clear answer.
4. Always mention the source document name and page number cited in the context.
"""


def load_env_credentials() -> dict:
    """
    Reads key-value pairs from .env file if present.
    """
    creds = {}
    env_paths = [".env", os.path.expanduser("~/.env")]
    for path in env_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            creds[k.strip()] = v.strip().strip("'\"")
            except Exception:
                pass
    return creds


def get_active_api_key() -> tuple:
    """
    Checks for available API keys: Gemini, Groq, or OpenAI.
    Returns (provider, api_key) or (None, None).
    """
    env_vars = load_env_credentials()
    
    # Priority 1: Gemini API Key
    gemini_key = os.environ.get("GEMINI_API_KEY") or env_vars.get("GEMINI_API_KEY")
    if gemini_key:
        return ("gemini", gemini_key)

    # Priority 2: Groq API Key
    groq_key = os.environ.get("GROQ_API_KEY") or env_vars.get("GROQ_API_KEY")
    if groq_key:
        return ("groq", groq_key)

    # Priority 3: OpenAI API Key
    openai_key = os.environ.get("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY")
    if openai_key:
        return ("openai", openai_key)

    return (None, None)


def call_gemini_api(prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """
    Calls Google Gemini REST API without requiring heavy external SDK packages.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        candidates = res_data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
    return "Information not found in the available documents."


def call_openai_compatible_api(prompt: str, api_key: str, endpoint: str, model: str) -> str:
    """
    Calls OpenAI / Groq REST API without requiring external packages.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        choices = res_data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "").strip()
    return "Information not found in the available documents."


def is_ollama_available() -> bool:
    """
    Checks if Ollama daemon is reachable on localhost:11434 with instant cache.
    """
    return check_ollama_online()


def build_rag_prompt(user_question: str, retrieved_context: str) -> str:
    """
    Constructs the grounded RAG prompt adhering to Step 18 of the specification.
    """
    if not retrieved_context.strip():
        retrieved_context = "[NO CONTEXT AVAILABLE]"

    prompt = f"""{RAG_SYSTEM_PROMPT}

CONTEXT:
{retrieved_context}

QUESTION:
{user_question}

ANSWER:"""
    return prompt


def generate_llm_answer(
    prompt: str,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1
) -> dict:
    """
    Answer Generation Engine:
    1. Checks if Cloud API Key is configured (Gemini / Groq / OpenAI).
    2. Else checks if local Ollama daemon is running.
    3. Else uses local zero-dependency extractive fallback.
    """
    # Option 1: Cloud API (if user supplied an API key)
    provider, api_key = get_active_api_key()
    if provider == "gemini" and api_key:
        try:
            ans = call_gemini_api(prompt, api_key)
            return {"answer": ans, "engine": "Google Gemini API (Cloud)", "status": "online"}
        except Exception as e:
            pass
    elif provider == "groq" and api_key:
        try:
            ans = call_openai_compatible_api(prompt, api_key, "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile")
            return {"answer": ans, "engine": "Groq Cloud API (Llama 3.3)", "status": "online"}
        except Exception:
            pass
    elif provider == "openai" and api_key:
        try:
            ans = call_openai_compatible_api(prompt, api_key, "https://api.openai.com/v1/chat/completions", "gpt-4o-mini")
            return {"answer": ans, "engine": "OpenAI API (GPT-4o-mini)", "status": "online"}
        except Exception:
            pass

    # Option 2: Local Ollama daemon
    if is_ollama_available():
        import ollama
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            raw_answer = response["message"]["content"].strip()
            return {"answer": raw_answer, "engine": f"Ollama ({model_name})", "status": "online"}
        except Exception:
            pass

    # Option 3: Intelligent Extractive Fallback
    return _generate_extractive_fallback(prompt)


def _generate_extractive_fallback(prompt: str) -> Dict[str, Any]:
    """
    Zero-dependency extractive fallback used when Ollama is offline or installing.
    Extracts relevant sentences from the context or safely returns the refusal phrase.
    """
    # Extract Context and Question from prompt
    context_match = re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:", prompt, re.DOTALL)
    question_match = re.search(r"QUESTION:\s*(.*?)\s*ANSWER:", prompt, re.DOTALL)

    context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""

    refusal_resp = {
        "answer": "Information not found in the available documents.",
        "engine": "Local Extractive Engine (Ollama Offline)",
        "status": "offline_fallback"
    }

    if not context or context == "[NO CONTEXT AVAILABLE]":
        return refusal_resp

    # Comprehensive stop words list
    stop_words = {
        "what", "is", "the", "for", "a", "an", "how", "much", "many", "can", "i", 
        "to", "in", "of", "and", "or", "are", "do", "does", "who", "will", "be", 
        "at", "by", "from", "on", "with", "as", "about", "there", "their", "this",
        "that", "these", "those", "college", "student", "students"
    }
    q_words = [w.lower() for w in re.findall(r"\w+", question) if w.lower() not in stop_words and len(w) > 2]

    if not q_words:
        return refusal_resp

    # Split context into paragraphs
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    best_para = ""
    best_matches = 0

    for para in paragraphs:
        # Check text body without header line
        lines = para.split("\n")
        body = "\n".join(lines[1:]) if (lines and lines[0].startswith("[Document")) else para
        body_words = set(re.findall(r"\w+", body.lower()))
        
        matches = sum(1 for w in q_words if w in body_words)
        if matches > best_matches:
            best_matches = matches
            best_para = para

    # Grounding & hallucination check:
    # Must match at least 2 key terms AND at least 50% of query keywords
    match_ratio = best_matches / max(1, len(q_words))
    if best_matches < 2 or match_ratio < 0.5:
        return refusal_resp

    # Extract source line if present
    source_info = "Refer to the cited document."
    lines = best_para.split("\n")
    if lines and lines[0].startswith("[Document"):
        source_info = lines[0].strip("[]")
        body_text = "\n".join(lines[1:]).strip()
    else:
        body_text = best_para

    answer_text = (
        f"According to the official college records ({source_info}):\n\n"
        f"{body_text}\n\n"
        f"*(Note: Ollama is currently offline; displaying verified retrieved context directly)*"
    )

    return {
        "answer": answer_text,
        "engine": "Local Extractive Engine (Ollama Offline)",
        "status": "offline_fallback"
    }


if __name__ == "__main__":
    print("Ollama online status:", is_ollama_available())
