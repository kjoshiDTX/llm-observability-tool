import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_obs import Tracer, trace, configure


class _Collector:
    """Minimal in-process stand-in for the collector's HTTP API."""

    def __init__(self):
        self.posts = []
        self.snapshot = [{"runtime": "ollama", "model": "qwen2.5:0.5b"}]
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.posts.append((self.path, json.loads(self.rfile.read(n))))
                self.send_response(201)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_GET(self):
                body = json.dumps(outer.snapshot).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def wait(self, n, timeout=3.0):
        deadline = time.time() + timeout
        while len(self.posts) < n and time.time() < deadline:
            time.sleep(0.01)
        return self.posts

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def collector():
    c = _Collector()
    yield c
    c.close()


def test_context_manager_sends_begin_and_trace(collector):
    tr = Tracer(url=collector.url, default_tags={"env": "test"})
    with tr.trace("summarize", model="qwen2.5:0.5b", tags={"version": "v1"}) as span:
        time.sleep(0.02)
        span.set_tokens(prompt=12, completion=30)
        span.mark_first_token()
    posts = collector.wait(2)
    assert [p[0] for p in posts] == ["/traces/begin", "/traces"]
    begin, rec = posts[0][1], posts[1][1]
    assert begin["id"] == rec["id"] == span.id
    assert begin["runtime"] == "ollama"  # resolved via GET /snapshot
    assert rec["name"] == "summarize"
    assert rec["model"] == "qwen2.5:0.5b"
    assert rec["tags"] == {"env": "test", "version": "v1"}
    assert rec["prompt_tokens"] == 12 and rec["completion_tokens"] == 30
    assert rec["duration_ms"] >= 20
    assert rec["ttft_ms"] > 0
    assert "error" not in rec
    assert rec["start_time"].endswith("Z")


def test_exception_is_recorded_and_reraised(collector):
    tr = Tracer(url=collector.url)
    with pytest.raises(ValueError):
        with tr.trace("boom", model="m", runtime="ollama"):
            raise ValueError("bad prompt")
    rec = collector.wait(2)[1][1]
    assert rec["error"].startswith("ValueError: bad prompt")


def test_decorator_auto_records_ollama_response(collector):
    tr = Tracer(url=collector.url)

    @tr.trace("gen", model="m", runtime="ollama")
    def call():
        return {"model": "m", "eval_count": 50, "prompt_eval_count": 8, "eval_duration": 500_000_000}

    assert call()["eval_count"] == 50
    rec = collector.wait(2)[1][1]
    assert rec["completion_tokens"] == 50 and rec["prompt_tokens"] == 8
    assert abs(rec["tokens_per_sec"] - 100.0) < 1e-6


def test_openai_shaped_response(collector):
    tr = Tracer(url=collector.url)
    with tr.trace("chat", model="Qwen/Qwen2.5-0.5B-Instruct", runtime="vllm") as span:
        span.record_openai_response({"model": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 9}})
    rec = collector.wait(2)[1][1]
    assert rec["prompt_tokens"] == 5 and rec["completion_tokens"] == 9 and rec["runtime"] == "vllm"


def test_collector_down_never_raises():
    tr = Tracer(url="http://127.0.0.1:1", timeout=0.2)
    with tr.trace("x", model="m", runtime="ollama"):
        pass
    assert tr.flush(3.0)
    assert tr.stats["failed"] >= 1
    assert tr.stats["sent"] == 0


def test_disabled_tracer_sends_nothing(collector):
    tr = Tracer(url=collector.url, enabled=False)
    with tr.trace("x", model="m") as span:
        pass
    assert span.start > 0
    time.sleep(0.1)
    assert collector.posts == []


def test_module_level_trace_uses_configured_default(collector):
    configure(url=collector.url)
    with trace("global", model="m", runtime="ollama"):
        pass
    assert len(collector.wait(2)) == 2


def test_span_to_dict_and_tag_coercion():
    tr = Tracer(url="http://127.0.0.1:1", enabled=False)
    with tr.trace("x", model="m", tags={"n": 3}) as span:
        span.set_tag("k", 1.5)
    d = span.to_dict()
    assert d["tags"] == {"n": "3", "k": "1.5"}
    assert d["duration_ms"] >= 0
