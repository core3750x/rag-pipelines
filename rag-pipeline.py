# rag-pipeline.py
# Open WebUI Pipelines (PIPE): Qdrant RAG -> Ollama (phi) with Ollama embeddings (nomic-embed-text)

import os
import json
import traceback
from typing import Any, Dict, List

import requests


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else str(v).strip()


QDRANT_URL = _env("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333").rstrip("/")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "docs_pdf")

OLLAMA_URL = _env("OLLAMA_URL", "http://ollama.ollama.svc.cluster.local:11434").rstrip("/")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL = _env("LLM_MODEL", "phi:latest")

TOP_K = int(_env("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(_env("MAX_CONTEXT_CHARS", "12000"))
DEBUG_RAG = (_env("DEBUG_RAG", "0") == "1")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "").lower() == "user":
            return (m.get("content") or "").strip()
    return ""


def _extract_text(payload: Any) -> str:
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

        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    if isinstance(payload, str):
        return payload.strip()

    return str(payload)


def _extract_source(payload: Any) -> str:
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


def ollama_embeddings(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        headers={"Content-Type": "application/json"},
        data=_dumps({"model": EMBED_MODEL, "prompt": text}),
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Bad embeddings response: {data}")
    return emb


def qdrant_search(vector: List[float]) -> List[Dict[str, Any]]:
    r = requests.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        headers={"Content-Type": "application/json"},
        data=_dumps({"vector": vector, "limit": TOP_K, "with_payload": True, "with_vectors": False}),
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    res = data.get("result")
    return res if isinstance(res, list) else []


def build_context(hits: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    total = 0

    for i, h in enumerate(hits, start=1):
        payload = h.get("payload", {})
        txt = _extract_text(payload)
        src = _extract_source(payload)
        score = h.get("score")

        header = f"[{i}]"
        if score is not None:
            header += f" score={score}"
        if src:
            header += f" source={src}"

        block = f"{header}\n{txt}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts).strip()


def ollama_chat(question: str, context: str) -> str:
    system = (
        "Ты отвечаешь строго по CONTEXT из внутренней документации.\n"
        "Если ответа в CONTEXT нет — честно скажи, что в документации это не найдено.\n"
        "Пиши по-русски. Коротко и по делу.\n"
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        headers={"Content-Type": "application/json"},
        data=_dumps(
            {
                "model": LLM_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ),
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Bad chat response: {data}")
    return content.strip()


class Pipeline:
    """
    ВАЖНО: Это PIPE-пайплайн. Pipelines-сервис ожидает .pipe(...)
    """

    def __init__(self) -> None:
        self.pipeline_id = "rag-pipeline"
        self.pipeline_name = "docs_pdf (Qdrant) → phi (RAG)"

    def pipes(self) -> List[Dict[str, str]]:
        # что увидит Open WebUI в списке моделей
        return [{"id": self.pipeline_id, "name": self.pipeline_name}]

    def pipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            messages = body.get("messages") or []
            question = _last_user_text(messages)

            if not question:
                return {"choices": [{"message": {"role": "assistant", "content": "Не вижу вопроса."}}]}

            vec = ollama_embeddings(question)
            hits = qdrant_search(vec)
            context = build_context(hits)
            answer = ollama_chat(question, context)

            if DEBUG_RAG:
                answer += "\n\n---\nDEBUG:\n" + _dumps(
                    {
                        "qdrant_url": QDRANT_URL,
                        "collection": QDRANT_COLLECTION,
                        "ollama_url": OLLAMA_URL,
                        "embed_model": EMBED_MODEL,
                        "llm_model": LLM_MODEL,
                        "hits": len(hits),
                    }
                )

            # Open WebUI ждёт OpenAI-like форму
            return {"choices": [{"message": {"role": "assistant", "content": answer}}]}

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if DEBUG_RAG:
                msg += "\n" + traceback.format_exc()
            return {"choices": [{"message": {"role": "assistant", "content": f"RAG pipeline error: {msg}"}}]}
