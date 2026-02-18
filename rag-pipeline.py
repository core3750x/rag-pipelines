# qdrant_rag_phi.py
# Open WebUI Pipelines: Qdrant RAG -> Ollama (phi) using Ollama embeddings (nomic-embed-text)
#
# Env:
#   QDRANT_URL            default: http://qdrant.qdrant.svc.cluster.local:6333
#   QDRANT_COLLECTION     default: docs_pdf
#   OLLAMA_URL            default: http://ollama.ollama.svc.cluster.local:11434
#   EMBED_MODEL           default: nomic-embed-text:latest
#   LLM_MODEL             default: phi:latest
#   TOP_K                 default: 6
#   MAX_CONTEXT_CHARS     default: 12000
#   CONTEXT_STRATEGY      default: concat   (concat|xml)
#   INCLUDE_SCORES        default: 1
#   INCLUDE_SOURCES       default: 1
#   REQUEST_TIMEOUT_SEC   default: 300
#   DEBUG_RAG             default: 0

import os
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else str(v).strip()


QDRANT_URL = _env("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333").rstrip("/")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "docs_pdf")

OLLAMA_URL = _env("OLLAMA_URL", "http://ollama.ollama.svc.cluster.local:11434").rstrip("/")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL = _env("LLM_MODEL", "phi:latest")

TOP_K = int(_env("TOP_K", "6") or "6")
MAX_CONTEXT_CHARS = int(_env("MAX_CONTEXT_CHARS", "12000") or "12000")

CONTEXT_STRATEGY = (_env("CONTEXT_STRATEGY", "concat") or "concat").lower()
INCLUDE_SCORES = (_env("INCLUDE_SCORES", "1") != "0")
INCLUDE_SOURCES = (_env("INCLUDE_SOURCES", "1") != "0")

REQUEST_TIMEOUT_SEC = int(_env("REQUEST_TIMEOUT_SEC", "300") or "300")
DEBUG_RAG = (_env("DEBUG_RAG", "0") == "1")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pick_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "").lower() == "user":
            return (m.get("content") or "").strip()
    return ""


def _extract_text_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        for k in ("text", "content", "chunk", "page_content", "body"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        meta = payload.get("metadata")
        if isinstance(meta, dict):
            for k in ("text", "content", "chunk", "page_content", "body"):
                v = meta.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # fallback: stringify payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    if isinstance(payload, str):
        return payload.strip()

    return str(payload)


def _extract_source_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for k in ("source", "file", "path", "s3_key", "url", "title", "doc"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for k in ("source", "file", "path", "s3_key", "url", "title", "doc"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def ollama_embeddings(prompt: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        headers={"Content-Type": "application/json"},
        data=_json_dumps({"model": EMBED_MODEL, "prompt": prompt}),
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Bad embeddings response: {data}")
    return emb


def qdrant_search(vector: List[float]) -> List[Dict[str, Any]]:
    payload = {
        "vector": vector,
        "limit": TOP_K,
        "with_payload": True,
        "with_vectors": False,
    }
    r = requests.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        headers={"Content-Type": "application/json"},
        data=_json_dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    data = r.json()
    res = data.get("result")
    return res if isinstance(res, list) else []


def _build_context(hits: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    total = 0

    for i, h in enumerate(hits, start=1):
        score = h.get("score")
        payload = h.get("payload", {})
        text = _extract_text_from_payload(payload)
        source = _extract_source_from_payload(payload)

        head_bits: List[str] = [f"[{i}]"]
        if INCLUDE_SCORES and score is not None:
            head_bits.append(f"score={score}")
        if INCLUDE_SOURCES and source:
            head_bits.append(f"source={source}")

        header = " ".join(head_bits)
        block = f"{header}\n{text}\n"

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

    context = "\n---\n".join(parts).strip()

    if CONTEXT_STRATEGY == "xml":
        # useful for some models to parse better
        return f"<context>\n{context}\n</context>" if context else "<context></context>"

    return context


def ollama_chat_with_context(question: str, context: str) -> str:
    system = (
        "Ты отвечаешь строго по CONTEXT из внутренней документации.\n"
        "Если ответа в CONTEXT нет — честно скажи, что в документации это не найдено.\n"
        "Пиши по-русски. Коротко и по делу.\n"
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        headers={"Content-Type": "application/json"},
        data=_json_dumps(
            {
                "model": LLM_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ),
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Bad chat response: {data}")
    return content.strip()


class Pipeline:
    # This name is shown as a model/provider in Open WebUI when using Pipelines.
    name = "docs_pdf (Qdrant) → phi (RAG)"

    def __init__(self) -> None:
        pass

    # OpenAI-compatible endpoint used by Pipelines service
    def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        started = _now_ms()
        try:
            messages = body.get("messages") or []
            question = _pick_last_user_message(messages)
            if not question:
                return {
                    "id": "rag-empty",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Не вижу вопроса."}}],
                }

            vec = ollama_embeddings(question)
            hits = qdrant_search(vec)
            context = _build_context(hits)
            answer = ollama_chat_with_context(question, context)

            if DEBUG_RAG:
                debug_info = {
                    "qdrant_hits": len(hits),
                    "collection": QDRANT_COLLECTION,
                    "embed_model": EMBED_MODEL,
                    "llm_model": LLM_MODEL,
                    "ms": _now_ms() - started,
                }
                answer = f"{answer}\n\n---\nDEBUG:\n{json.dumps(debug_info, ensure_ascii=False, indent=2)}"

            return {
                "id": "rag",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
            }

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if DEBUG_RAG:
                err = err + "\n" + traceback.format_exc()

            return {
                "id": "rag-error",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"RAG pipeline error: {err}"}}],
            }
