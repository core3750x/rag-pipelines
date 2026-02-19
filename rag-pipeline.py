import os
import json
import time
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests


# -----------------------------
# helpers
# -----------------------------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return "" if v is None else str(v)


def as_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def as_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def now_ts() -> int:
    return int(time.time())


def truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def pick_user_prompt(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    if messages:
        c = messages[-1].get("content")
        if isinstance(c, str):
            return c.strip()
    return ""


def is_title_generation(body: Dict[str, Any]) -> bool:
    md = body.get("metadata") or {}
    if isinstance(md, dict) and md.get("task") == "title_generation":
        return True

    messages = body.get("messages") or []
    if isinstance(messages, list) and messages:
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str) and "Create a concise, 3-5 word title" in content:
            return True
    return False


def make_title(prompt: str) -> str:
    p = (prompt or "").strip()
    p = re.sub(r"\s+", " ", p)

    # если это "Prompt: ...."
    m = re.search(r"Prompt:\s*(.*)$", p, flags=re.IGNORECASE)
    if m:
        p = m.group(1).strip()

    # уберём мусорные хвосты
    p = p.strip().strip('"').strip("'")
    if not p:
        return "Новый чат"

    # первые 3–6 слов
    words = [w for w in re.split(r"\s+", p) if w]
    title = " ".join(words[:6])
    title = title.replace('"', "").replace("'", "").strip()
    return title or "Новый чат"


def openai_response_text(model_id: str, content: str) -> Dict[str, Any]:
    t = now_ts()
    return {
        "id": f"chatcmpl-{t}",
        "object": "chat.completion",
        "created": t,
        "model": model_id,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def stream_as_openai_chunks(model_id: str, full_text: str, chunk_size: int = 80) -> Generator[Dict[str, Any], None, None]:
    created = now_ts()
    # first chunk: role
    yield {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }

    i = 0
    while i < len(full_text):
        part = full_text[i : i + chunk_size]
        i += chunk_size
        yield {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }

    # final chunk
    yield {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


# -----------------------------
# Qdrant + Ollama clients
# -----------------------------
def ollama_embeddings(ollama_url: str, model: str, text: str) -> List[float]:
    url = ollama_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embeddings response: {data}")
    return emb


def qdrant_search(qdrant_url: str, collection: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
    url = qdrant_url.rstrip("/") + f"/collections/{collection}/points/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("result", []) or []


def ollama_chat(ollama_url: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    msg = (data.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise RuntimeError(f"Unexpected chat response: {data}")
    return msg


# -----------------------------
# payload parsing + pretty output
# -----------------------------
def payload_text(payload: Dict[str, Any]) -> str:
    # максимально терпимо к разным инжестерам
    for k in ("text", "content", "chunk", "page_text", "body"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def payload_source(payload: Dict[str, Any]) -> str:
    for k in ("source", "file", "filename", "document", "title", "path", "url"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def payload_page(payload: Dict[str, Any]) -> Optional[Any]:
    for k in ("page", "page_number", "pageno"):
        v = payload.get(k)
        if v is not None:
            return v
    return None


def build_context_and_cards(points: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, str]:
    """
    Возвращаем:
      - context для LLM (плотный)
      - markdown карточки "Найдено в документации"
    """
    ctx_parts: List[str] = []
    cards: List[str] = []

    for idx, p in enumerate(points, start=1):
        payload = p.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        txt = payload_text(payload)
        src = payload_source(payload)
        page = payload_page(payload)

        score = p.get("score")
        score_s = ""
        if isinstance(score, (int, float)):
            score_s = f" (score={score:.4f})"

        if not txt:
            continue

        # карточка: короткий сниппет
        snippet = re.sub(r"\s+", " ", txt).strip()
        snippet = truncate(snippet, 420)

        where = src
        if page is not None:
            where = f"{src} · стр. {page}"

        cards.append(f"**{idx}. {where}**{score_s}\n\n> {snippet}\n")

        # контекст: чуть больше
        ctx_parts.append(f"[{idx}] {where}\n{txt.strip()}")

    context = "\n\n---\n\n".join(ctx_parts)
    context = truncate(context, max_context_chars)

    cards_md = ""
    if cards:
        cards_md = "## Найдено в документации\n\n" + "\n\n".join(cards) + "\n"
    return context, cards_md


# -----------------------------
# Pipeline
# -----------------------------
class Pipeline:
    """
    Open WebUI Pipelines:
      - class Pipeline
      - method pipe(self, body: dict, user: dict | None)
    """

    def __init__(self) -> None:
        self.id = "rag-pipeline"
        self.name = "rag-pipeline"

        self.debug = as_bool(env("DEBUG_RAG", "0"), False)

        self.qdrant_url = env("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333")
        self.qdrant_collection = env("QDRANT_COLLECTION", "docs_pdf")

        self.ollama_url = env("OLLAMA_URL", "http://ollama.ollama.svc.cluster.local:11434")
        self.embed_model = env("EMBED_MODEL", "nomic-embed-text:latest")
        self.llm_model = env("LLM_MODEL", "qwen3:1.7b")

        self.top_k = as_int(env("TOP_K", "5"), 5)
        self.max_context_chars = as_int(env("MAX_CONTEXT_CHARS", "8000"), 8000)

        # чтобы не было каши, если Open WebUI дёргает несколько раз подряд
        self._cache: Dict[str, Tuple[float, str]] = {}  # key -> (ts, full_answer)
        self._cache_ttl_sec = 60.0

    def _log(self, *args: Any) -> None:
        if self.debug:
            print("DEBUG_RAG:", *args, flush=True)

    def _cache_get(self, key: str) -> Optional[str]:
        v = self._cache.get(key)
        if not v:
            return None
        ts, ans = v
        if time.time() - ts > self._cache_ttl_sec:
            self._cache.pop(key, None)
            return None
        return ans

    def _cache_set(self, key: str, ans: str) -> None:
        self._cache[key] = (time.time(), ans)

    def pipe(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Any:
        stream = bool(body.get("stream", True))
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            messages = []

        # 1) Заголовок чата — не RAG, не Ollama, не Qdrant
        if is_title_generation(body):
            prompt = pick_user_prompt(messages)
            title = make_title(prompt)
            return openai_response_text(self.id, title)

        # 2) Нормальный запрос
        prompt = pick_user_prompt(messages)
        if not prompt:
            return openai_response_text(self.id, "Напиши запрос 🙂")

        # key для дедупликации (UI может дернуть 2-5 раз подряд)
        chat_id = body.get("chat_id") or ((body.get("metadata") or {}) if isinstance(body.get("metadata"), dict) else {}).get("chat_id")
        cache_key = f"{chat_id or 'nochat'}::{prompt}"

        cached = self._cache_get(cache_key)
        if cached:
            if stream:
                return stream_as_openai_chunks(self.id, cached)
            return openai_response_text(self.id, cached)

        # 2.1) Embeddings
        try:
            qvec = ollama_embeddings(self.ollama_url, self.embed_model, prompt)
        except Exception as e:
            msg = f"Не смог сделать эмбеддинг через Ollama ({self.embed_model}): {e}"
            if stream:
                return stream_as_openai_chunks(self.id, msg)
            return openai_response_text(self.id, msg)

        # 2.2) Qdrant search
        try:
            points = qdrant_search(self.qdrant_url, self.qdrant_collection, qvec, self.top_k)
        except Exception as e:
            msg = f"Не смог искать в Qdrant ({self.qdrant_collection}): {e}"
            if stream:
                return stream_as_openai_chunks(self.id, msg)
            return openai_response_text(self.id, msg)

        context, cards_md = build_context_and_cards(points, self.max_context_chars)

        # если ничего не нашли — честно говорим и всё
        if not context.strip():
            answer = (
                f"{cards_md}\n"
                "## Ответ\n\n"
                "В базе знаний не нашёл релевантных фрагментов по запросу. "
                "Попробуй переформулировать (например: `CI/CD GitLab Runner cache S3`, `ArgoCD sync hooks`, `Ingress TLS default cert`)."
            ).strip()
            self._cache_set(cache_key, answer)
            if stream:
                return stream_as_openai_chunks(self.id, answer)
            return openai_response_text(self.id, answer)

        # 2.3) LLM ответ (строго по контексту)
        system = (
            "Ты ассистент по внутренней документации компании.\n"
            "Отвечай по-русски.\n"
            "Используй только предоставленный контекст. Если ответа нет — скажи, что в документации не найдено.\n"
            "Пиши структурировано: кратко, затем шаги/команды.\n"
            "Не выдумывай факты.\n"
        )

        user_msg = (
            f"Запрос пользователя:\n{prompt}\n\n"
            f"Контекст из базы знаний (фрагменты):\n{context}\n\n"
            "Сформируй ответ. Если есть конкретные команды/пути/настройки — приведи их."
        )

        try:
            llm_text = ollama_chat(
                self.ollama_url,
                self.llm_model,
                [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            ).strip()
        except Exception as e:
            msg = f"Не смог получить ответ от Ollama ({self.llm_model}): {e}"
            if stream:
                return stream_as_openai_chunks(self.id, msg)
            return openai_response_text(self.id, msg)

        answer = (f"{cards_md}\n## Ответ\n\n{llm_text}").strip()
        self._cache_set(cache_key, answer)

        # Важно: чтобы OpenWebUI не плодил два вызова Ollama — мы стримим уже готовый текст
        if stream:
            return stream_as_openai_chunks(self.id, answer)
        return openai_response_text(self.id, answer)
