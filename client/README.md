# llm-obs (Python client)

Tracing client for the llm-obs collector. See the repository README for full docs.

```python
from llm_obs import trace

with trace(name="summarize_endpoint", model="llama3.2:3b") as span:
    resp = ollama.generate(...)
    span.record_ollama_response(resp)
```
