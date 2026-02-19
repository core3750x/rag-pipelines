# rag-pipeline.py
# Open WebUI Pipelines (PIPE): Qdrant RAG -> Ollama (LLM) + Ollama embeddings (nomic-embed-text)
#
# Key points:
# - Supports OpenAI-compatible streaming (SSE) when Open WebUI sends stream=true
# - Handles Open WebUI metadata task "title_generation" locally (no Qdrant, no LLM)
# - Trims embedding input to avoid nomic context warnings (n_ctx_train=2048)
#
# Env vars expected:
#   QDRANT_URL         (default: http://qdrant.qdrant.svc.cluster.local:6333)
#   QDRANT_COLLECTION  (default: docs_pdf)
#   OLLAMA_URL         (default: http://ollama.ollama.svc.cluster.local:11434)
#   EMBED_MODEL        (default: nomic-embed-text:latest)
#   LLM_MODEL          (default: qwen3:1.7b)   # поменяй на то, что реально есть в ollama
#   TOP_K              (default: 3)
#   MAX_CONTEXT_CHARS  (default: 6000)
#   EMBED_MAX_CHARS    (default: 1800)
#   DEBUG_RAG          (default: 0)            # 1 => добавляет debug и трейсбек в ошибки

import os
import re
import json
import time
import uuid
import traceback
from typing import Any, Dict, Iterable, List, Optional

import requests


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else str(v).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _safe_truncate(s: str, max_chars: int) -> str:
    s = _clean_ws(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _sse(data: Any) -> str:
    # OpenAI streaming expects SSE lines: "data: {...}\n\n"
    return f"data: {_dumps(data)}\n\n"


def _extract_prompt_from_titlegen_blob(text: str) -> str:
    # Open WebUI title_generation: "... Prompt: <original>"
    m = re.search(r"Prompt:\s*(.+)\s*$", text or "", flags=re.IGNORECASE | re.DOTALL)
    if m:
        return _clean_ws(m.group(1))
    return _clean_ws(text or "")


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") or "").lower() == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
    return ""


def _get_task(kwargs: Dict[str, Any]) -> str:
    body = kwargs.get("body")
    if isinstance(body, dict):
        md = body.get("metadata")
        if isinstance(md, dict) and isinstance(md.get("task"), str):
            return md.get("task")  # type: ignore[return-value]
    md2 = kwargs.get("metadata")
    if isinstance(md2, dict) and isinstance(md2.get("task"), str):
        return md2.get("task")  # type: ignore[return-value]
    return ""


def _get_stream_flag(kwargs: Dict[str, Any]) -> bool:
    # Open WebUI usually sends stream=true
    if isinstance(kwargs.get("stream"), bool):
        return bool(kwargs.get("stream"))
    body = kwargs.get("body")
    if isinstance(body, dict) and isinstance(body.get("stream"), bool):
        return bool(body.get("stream"))
    return False


def _extract_question(kwargs: Dict[str, Any]) -> str:
    # Different Open WebUI versions call pipe with different shapes
    um = kwargs.get("user_message")
    if isinstance(um, str) and um.strip():
        return um.strip()

    body = kwargs.get("body")
    if isinstance(body, dict):
        msgs = body.get("messages")
        if isinstance(msgs, list):
            q = _last_user_text(msgs)
            if q:
                return q
        pr = body.get("prompt")
        if isinstance(pr, str) and pr.strip():
            return pr.strip()

    msgs2 = kwargs.get("messages")
    if isinstance(msgs2, list):
        q = _last_user_text(msgs2)
        if q:
            return q

    prompt = kwargs.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    return ""


def _make_title(prompt_blob: str) -> str:
    prompt = _extract_prompt_from_titlegen_blob(prompt_blob)
    words = re.findall(r"[A-Za-zА-Яа-я0-9]+(?:-[A-Za-zА-Яа-я0-9]+)*", prompt)
    base = " ".join(words[:5]).strip()
    if not base:
        return "💬 Новый чат"
    base = base[0].upper() + base[1:]
    return f"💬 {base}"


def _extract_text_from_payload(payload: Any) -> str:
    # Try common keys produced by different ingesters
    if isinstance(payload, dict):
        for k in ("text", "content", "chunk", "page_content", "body"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        md = payload.get("metadata")
        if isinstance(md, dict):
            for k in ("text", "content", "chunk", "page_content", "body"):
                v = md.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return ""


# -----------------------------
# Config
# -----------------------------
QDRANT_URL = _env("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333").rstrip("/")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "docs_pdf")

OLLAMA_URL = _env("OLLAMA_URL", "http://ollama.ollama.svc.cluster.local:11434").rstrip("/")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL = _env("LLM_MODEL", "qwen3:1.7b")

TOP_K = _env_int("TOP_K", 3)
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 6000)
EMBED_MAX_CHARS = _env_int("EMBED_MAX_CHARS", 1800)

DEBUG_RAG = (_env("DEBUG_RAG", "0") == "1")


# -----------------------------
# Qdrant + Ollama
# -----------------------------
def ollama_embeddings(text: str) -> List[float]:
    text = _extract_prompt_from_titlegen_blob(text)
    text = _safe_truncate(text, EMBED_MAX_CHARS)

    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        headers={"Content-Type": "application/json"},
        data=_dumps({"model": EMBED_MODEL, "prompt": text}),
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Bad embeddings response: {data}")
    return emb


def qdrant_search(vector: List[float], limit: int) -> List[Dict[str, Any]]:
    payload = {
        "vector": vector,
        "limit": int(limit),
        "with_payload": True,
        "with_vectors": False,
    }
    r = requests.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
        headers={"Content-Type": "application/json"},
        data=_dumps(payload),
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    res = data.get("result")
    return res if isinstance(res, list) else []


def build_context(hits: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    total = 0

    for h in hits or []:
        payload = h.get("payload", {})
        txt = _extract_text_from_payload(payload)
        if not txt:
            continue
        txt = _clean_ws(txt)
        if not txt:
            continue

        block = txt + "\n"
        if total + len(block) > max_chars:
            rem = max_chars - total
            if rem > 50:
                parts.append(block[:rem])
            break

        parts.append(block)
        total += len(block)

    context = "\n---\n".join([p.strip() for p in parts if p.strip()]).strip()
    return context


def ollama_chat_ru(question: str, context: str) -> str:
    system = (
        "Ты помощник по внутренней документации.\n"
        "Всегда отвечай на русском.\n"
        "Используй CONTEXT ниже как источник истины.\n"
        "Если ответа в CONTEXT нет — скажи: 'В документации не нашёл.' и предложи, что уточнить.\n"
        "Отвечай кратко и по делу.\n"
    )

    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    req = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        headers={"Content-Type": "application/json"},
        data=_dumps(req),
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Bad chat response: {data}")
    return content.strip()


# -----------------------------
# OpenAI-compatible response builders
# -----------------------------
def _openai_final(model: str, content: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _openai_stream_chunks(model: str, content: str) -> Iterable[str]:
    # Split into small chunks to look "live"
    chunk_size = 120
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"

    # First chunk can be empty delta (some clients like it)
    first = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield _sse(first)

    for i in range(0, len(content), chunk_size):
        piece = content[i : i + chunk_size]
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield _sse(chunk)

    last = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield _sse(last)
    yield "data: [DONE]\n\n"


# -----------------------------
# Pipeline class (required by Open WebUI Pipelines)
# -----------------------------
class Pipeline:
    def __init__(self) -> None:
        self.pipeline_id = "rag-pipeline"
        self.pipeline_name = "rag-pipeline"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": self.pipeline_id, "name": self.pipeline_name}]

    def pipe(self, *args: Any, **kwargs: Any) -> Any:
        started = time.time()
        try:
            task = _get_task(kwargs)
            stream = _get_stream_flag(kwargs)
            raw_q = _extract_question(kwargs)

            # 1) Title generation (metadata task) — local answer, no LLM/Qdrant
            if task == "title_generation":
                title = _make_title(raw_q)
                # For title_generation Open WebUI expects non-stream
                return _openai_final(self.pipeline_id, title)

            # 2) RAG
            question = _extract_prompt_from_titlegen_blob(raw_q)
            if not question:
                out = "Не вижу вопроса."
                return _openai_stream_chunks(self.pipeline_id, out) if stream else _openai_final(self.pipeline_id, out)

            vec = ollama_embeddings(question)
            hits = qdrant_search(vec, TOP_K)
            context = build_context(hits, MAX_CONTEXT_CHARS)

            if not context:
                answer = (
                    "В документации не нашёл.\n"
                    "Подскажи: это про Airflow Helm values, про connection в UI, или про переменные окружения в pod?"
                )
            else:
                answer = ollama_chat_ru(question, context)

            if DEBUG_RAG:
                took = round(time.time() - started, 3)
                answer += "\n\n---\nDEBUG:\n" + _dumps(
                    {
                        "took_s": took,
                        "llm_model": LLM_MODEL,
                        "embed_model": EMBED_MODEL,
                        "qdrant_url": QDRANT_URL,
                        "collection": QDRANT_COLLECTION,
                        "top_k": TOP_K,
                        "context_chars": len(context),
                        "hits": len(hits),
                    }
                )

            return _openai_stream_chunks(self.pipeline_id, answer) if stream else _openai_final(self.pipeline_id, answer)

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if DEBUG_RAG:
                msg += "\n" + traceback.format_exc()
            out = f"RAG pipeline error: {msg}"
            # Even on errors, respect stream flag so UI doesn’t ломается
            stream = _get_stream_flag(kwargs)
            return _openai_stream_chunks(self.pipeline_id, out) if stream else _openai_final(self.pipeline_id, out)
