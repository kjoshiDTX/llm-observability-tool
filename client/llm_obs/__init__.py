"""llm-obs: request-level tracing for local LLM apps (Ollama, vLLM).

    from llm_obs import trace

    with trace(name="summarize_endpoint", model="llama3.2:3b") as span:
        resp = call_ollama(prompt)
        span.record_ollama_response(resp)
"""

from .tracer import Span, Tracer, configure, get_tracer, trace

__version__ = "0.1.0"
__all__ = ["trace", "Tracer", "Span", "configure", "get_tracer", "__version__"]
