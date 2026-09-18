"""Request-level tracing for local LLM calls.

Usage::

    from llm_obs import trace

    with trace(name="summarize_endpoint", model="llama3.2:3b") as span:
        resp = ollama.generate(model="llama3.2:3b", prompt=prompt)
        span.record_ollama_response(resp)

or as a decorator::

    @trace(name="chat_turn", model="llama3.2:3b", tags={"version": "v2"})
    def answer(prompt): ...

Traces are shipped to the collector on a background thread; if the collector
is down the application is never slowed or broken.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

__all__ = ["Tracer", "Span", "trace", "configure", "get_tracer"]

_log = logging.getLogger("llm_obs")

DEFAULT_URL = "http://localhost:8765"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class Span:
    """A single traced LLM call. Created by :meth:`Tracer.trace`."""

    __slots__ = (
        "id", "name", "model", "runtime", "tags", "start", "end", "first_token_at",
        "prompt_tokens", "completion_tokens", "tokens_per_sec", "error", "_tracer",
    )

    def __init__(self, tracer: "Tracer", name: str, model: Optional[str], runtime: Optional[str],
                 tags: Optional[Dict[str, str]]):
        self._tracer = tracer
        self.id = uuid.uuid4().hex
        self.name = name
        self.model = model or ""
        self.runtime = runtime or ""
        self.tags: Dict[str, str] = dict(tracer.default_tags)
        if tags:
            self.tags.update({str(k): str(v) for k, v in tags.items()})
        self.start = 0.0
        self.end = 0.0
        self.first_token_at = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokens_per_sec = 0.0
        self.error = ""

    # -- user-facing helpers -------------------------------------------------
    def set_model(self, model: str, runtime: Optional[str] = None) -> "Span":
        self.model = model
        if runtime:
            self.runtime = runtime
        return self

    def set_tag(self, key: str, value: Any) -> "Span":
        self.tags[str(key)] = str(value)
        return self

    def set_tokens(self, prompt: Optional[int] = None, completion: Optional[int] = None) -> "Span":
        if prompt is not None:
            self.prompt_tokens = int(prompt)
        if completion is not None:
            self.completion_tokens = int(completion)
        return self

    def mark_first_token(self) -> "Span":
        """Call when the first streamed token arrives (records TTFT)."""
        if not self.first_token_at:
            self.first_token_at = time.time()
        return self

    def record_ollama_response(self, resp: Any) -> "Span":
        """Extract token counts / throughput from an Ollama generate or chat
        response (dict or object with attributes)."""
        g = _getter(resp)
        self.set_tokens(g("prompt_eval_count"), g("eval_count"))
        eval_dur = g("eval_duration")  # nanoseconds
        if eval_dur and self.completion_tokens:
            self.tokens_per_sec = self.completion_tokens / (float(eval_dur) / 1e9)
        model = g("model")
        if model and not self.model:
            self.model = str(model)
        if not self.runtime:
            self.runtime = "ollama"
        return self

    def record_openai_response(self, resp: Any) -> "Span":
        """Extract usage from an OpenAI-compatible response (vLLM, etc.)."""
        g = _getter(resp)
        usage = g("usage")
        if usage is not None:
            ug = _getter(usage)
            self.set_tokens(ug("prompt_tokens"), ug("completion_tokens"))
        model = g("model")
        if model and not self.model:
            self.model = str(model)
        return self

    # -- lifecycle -------------------------------------------------------------
    def _begin(self) -> None:
        self.start = time.time()
        if not self.runtime and self.model:
            self.runtime = self._tracer.runtime_for(self.model) or ""
        self._tracer._enqueue("/traces/begin", {
            "id": self.id, "name": self.name, "model": self.model,
            "runtime": self.runtime, "start_time": _iso(self.start),
        })

    def _finish(self, exc: Optional[BaseException] = None) -> None:
        self.end = time.time()
        if exc is not None and not self.error:
            self.error = f"{type(exc).__name__}: {exc}"[:500]
        self._tracer._enqueue("/traces", self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "runtime": self.runtime,
            "tags": self.tags,
            "start_time": _iso(self.start),
            "end_time": _iso(self.end or time.time()),
            "duration_ms": ((self.end or time.time()) - self.start) * 1000.0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_sec": self.tokens_per_sec,
        }
        if self.first_token_at:
            d["ttft_ms"] = (self.first_token_at - self.start) * 1000.0
        if self.error:
            d["error"] = self.error
        return d


def _getter(obj: Any) -> Callable[[str], Any]:
    if isinstance(obj, dict):
        return obj.get
    return lambda k: getattr(obj, k, None)


class _TraceContext:
    """Returned by Tracer.trace(); works as a context manager or decorator."""

    def __init__(self, tracer: "Tracer", name: str, model: Optional[str], runtime: Optional[str],
                 tags: Optional[Dict[str, str]]):
        self._tracer, self._name, self._model, self._runtime, self._tags = tracer, name, model, runtime, tags
        self._span: Optional[Span] = None

    def __enter__(self) -> Span:
        self._span = Span(self._tracer, self._name, self._model, self._runtime, self._tags)
        if self._tracer.enabled:
            self._span._begin()
        else:
            self._span.start = time.time()
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._span is not None and self._tracer.enabled:
            self._span._finish(exc)
        return False  # never swallow exceptions

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _TraceContext(self._tracer, self._name or fn.__name__, self._model, self._runtime, self._tags) as span:
                result = fn(*args, **kwargs)
                _auto_record(span, result)
                return result
        return wrapper


def _auto_record(span: Span, result: Any) -> None:
    """Best effort: recognise Ollama / OpenAI-shaped return values."""
    try:
        g = _getter(result)
        if g("eval_count") is not None or g("eval_duration") is not None:
            span.record_ollama_response(result)
        elif g("usage") is not None:
            span.record_openai_response(result)
    except Exception:  # pragma: no cover - purely defensive
        pass


class Tracer:
    """Ships spans to a collector. One instance per process is typical."""

    def __init__(self, url: Optional[str] = None, enabled: Optional[bool] = None,
                 timeout: float = 1.0, default_tags: Optional[Dict[str, str]] = None,
                 max_queue: int = 10000):
        self.url = (url or os.environ.get("LLM_OBS_URL") or DEFAULT_URL).rstrip("/")
        if enabled is None:
            enabled = os.environ.get("LLM_OBS_DISABLED", "") not in ("1", "true", "yes")
        self.enabled = enabled
        self.timeout = timeout
        self.default_tags: Dict[str, str] = {str(k): str(v) for k, v in (default_tags or {}).items()}
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._sent = 0
        self._failed = 0
        self._warned = False
        self._state_cache: tuple = (0.0, None)
        self._worker = threading.Thread(target=self._run, name="llm-obs-sender", daemon=True)
        self._worker.start()
        atexit.register(self.flush, 2.0)

    # -- public API -----------------------------------------------------------
    def trace(self, name: str, model: Optional[str] = None, runtime: Optional[str] = None,
              tags: Optional[Dict[str, str]] = None) -> _TraceContext:
        return _TraceContext(self, name, model, runtime, tags)

    def state(self, max_age: float = 2.0) -> Optional[list]:
        """Current runtime snapshots from the collector (cached briefly)."""
        ts, cached = self._state_cache
        if cached is not None and time.time() - ts < max_age:
            return cached
        data = self._get("/snapshot")
        if data is not None:
            self._state_cache = (time.time(), data)
        return data

    def runtime_for(self, model: str) -> Optional[str]:
        """Which runtime currently has `model` loaded, per the collector."""
        snaps = self.state()
        if not snaps:
            return None
        for s in snaps:
            if s.get("model") == model:
                return s.get("runtime")
        return None

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until queued records are sent (or timeout). Returns True if drained."""
        deadline = time.time() + timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.01)
        return self._q.empty()

    @property
    def stats(self) -> Dict[str, int]:
        return {"sent": self._sent, "failed": self._failed, "dropped": self._dropped, "queued": self._q.qsize()}

    # -- transport --------------------------------------------------------------
    def _enqueue(self, path: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait((path, payload))
        except queue.Full:
            self._dropped += 1

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, payload = item
            try:
                self._post(path, payload)
                self._sent += 1
            except Exception as e:  # collector down, etc. Never propagate.
                self._failed += 1
                if not self._warned:
                    self._warned = True
                    _log.warning("llm-obs: could not reach collector at %s (%s); traces will be dropped silently", self.url, e)
            finally:
                self._q.task_done()

    def _post(self, path: str, payload: Dict[str, Any]) -> None:
        import urllib.request
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()

    def _get(self, path: str) -> Optional[Any]:
        import urllib.request
        try:
            with urllib.request.urlopen(self.url + path, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


_default: Optional[Tracer] = None
_default_lock = threading.Lock()


def configure(**kwargs: Any) -> Tracer:
    """Replace the process-wide default tracer (url=, enabled=, default_tags=...)."""
    global _default
    with _default_lock:
        _default = Tracer(**kwargs)
        return _default


def get_tracer() -> Tracer:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Tracer()
    return _default


def trace(name: str, model: Optional[str] = None, runtime: Optional[str] = None,
          tags: Optional[Dict[str, str]] = None) -> _TraceContext:
    """Trace one LLM call with the default tracer (context manager or decorator)."""
    return get_tracer().trace(name, model=model, runtime=runtime, tags=tags)
