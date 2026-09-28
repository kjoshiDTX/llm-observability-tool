# llm-obs

An observability tool for developers building apps on **local** LLMs
(Ollama, vLLM). It is not a system-metrics dashboard. It exists to answer
three questions a developer actually has, in this order of importance:

1. **"Will this model fit before I load it?"** -- pre-flight capacity check
2. **"Why did this specific call get slow?"** -- request-level tracing plus
   a plain-language causal diagnosis
3. **"Did switching quantization/runtime/version make things better or
   worse?"** -- A/B regression comparison

Raw system metrics (memory, tokens/sec, KV-cache %) are internal plumbing
that feeds those three features. They are never the headline output.
Prometheus/Grafana export exists as an optional integration for teams that
already run that stack, not as the primary way to use this tool.

## The problem

Today, if a local-LLM app gets slow, a developer has no visibility into
*why*. They can watch Activity Monitor and guess. If they try to load a
model that's too big, they find out by OOMing. If they change a quantization
setting, they have no record of whether it actually helped. `llm-obs` turns
each of those into a direct answer instead of a guess:

```
$ llm-obs diagnose --slowest --name summarize_endpoint
CRITICAL  summarize_endpoint  model=qwen2.5:0.5b runtime=ollama  1.26s at 149.0 tok/s

Slow: 1264 ms - cold start (model just loaded). The model was loaded 0s before this call started, so
this request paid the load + warm-up cost (weights read from disk, graph compiled, KV cache
allocated).

Suggestion (cold_start): Keep the model resident (Ollama: keep_alive=-1) or send a warm-up request after load.
```

That is real output from this tool against a real local Ollama instance
(see [RESULTS.md](RESULTS.md) for the full session and how the diagnoser was
validated end to end).

## Architecture

```
                                   ┌─────────────────────────┐
  Your Python app                 │   llm-obs-collector (Go) │
  ┌────────────────────┐          │                          │
  │ with trace(name=...)│  HTTP   │  ┌────────┐  ┌─────────┐ │
  │   response = ...     │───────▶│  │  API   │─▶│  trace  │ │        ┌──────────┐
  │   span.record_...()  │        │  └────────┘  │ ingest  │─┼───────▶│  SQLite  │
  └────────────────────┘          │       ▲      └─────────┘ │        └──────────┘
                                   │       │            ▲     │             ▲
                                   │  ┌────┴─────┐      │     │             │
   Ollama  /api/ps ◀───────poll───┼──┤ adapters │      │     │             │
   vLLM    /metrics ◀──────poll───┼──┤ (ollama, │──────┘     │             │
                                   │  │  vllm)   │  ring      │             │
                                   │  └──────────┘  buffers   │             │
                                   │       │                  │             │
                                   │       ▼                  │             │
                                   │  ┌──────────┐  ┌────────┐│             │
                                   │  │ diagnoser│  │compare ││   optional  │
                                   │  │(plain-   │  │(A/B)   ││   ┌───────┐ │
                                   │  │ language)│  └────────┘├──▶│/metrics│─┼──▶ Prometheus ──▶ Grafana
                                   │  └──────────┘  ┌────────┐│   └───────┘ │    (docker-compose,
                                   │                │preflight│             │     Phase 8, optional)
                                   │                │ (check) │             │
                                   │                └────────┘│             │
                                   └─────────────────────────┘
                                              ▲
                                   ┌──────────┴──────────┐
                                   │  llm-obs CLI (Go)    │
                                   │  status / traces /   │
                                   │  diagnose / compare / │
                                   │  check / events       │
                                   └───────────────────────┘
```

- **Collector** (`collector/`, Go, single binary): polls Ollama and vLLM on
  independent goroutines into per-(runtime, model) ring buffers, ingests
  traces posted by the client SDK and joins them against the system state
  observed during their time window, stores everything in local SQLite, and
  serves a local HTTP API.
- **Client** (`client/`, Python, `pip install -e client/` as `llm_obs`): a
  context manager / decorator apps wrap around their own LLM calls. Ships
  traces to the collector on a background thread -- if the collector is
  down, the app is never slowed or broken.
- **CLI** (`collector/cmd/llm-obs`, built as `llm-obs`): the terminal
  front-end to everything above.

## Why tracing + diagnosis, not system metrics, is the core feature

A dashboard of tokens/sec and memory usage tells you *that* something is
wrong. It doesn't tell you *why*, and it doesn't remember which specific
request was slow or what the system was doing at that exact moment. The
[`diagnose`](collector/internal/core/diagnose/diagnoser.go) package exists
to close that gap: every trace is joined against the runtime snapshots,
events, and concurrent traces observed during its own time window, and a
rule engine attributes a slowdown to one of six causes (context/KV-cache
near limit, GPU-to-CPU backend fallback, cold start, contention, memory
pressure, or an unexplained regression against the request's own historical
baseline) with a plain-language explanation and a concrete suggestion. This
is deliberately the single most invested-in piece of the whole project --
see the 38 tests across
[`diagnoser_test.go`](collector/internal/core/diagnose/diagnoser_test.go) and
[`compare_test.go`](collector/internal/core/diagnose/compare_test.go)
covering every cause and proving it does not false-positive on healthy,
noisy-but-fine data.

Two real bugs in this logic were only found by running it against a live
Ollama instance rather than trusting unit tests alone (both are now fixed,
with regression tests, and documented in the commit history and
[RESULTS.md](RESULTS.md)): overlap detection at millisecond timestamp
resolution was misreporting purely sequential back-to-back calls as
concurrent, and cold-start attribution was crediting every call within its
15-second lookback window to one load event instead of only the first call
after it.

## Apple Silicon and the vLLM CPU-mode limitation

This was built and tested on an Apple Silicon Mac. Two things follow from
that:

- **Unified memory, not discrete VRAM.** The schema uses `MemUsedMB` /
  `MemTotalMB` with a `UnifiedMemory bool` rather than `VRAMUsedMB`, and the
  pre-flight checker caps the accelerator's usable budget at 75% of total
  RAM (Metal's working-set limit) rather than treating all "available"
  memory as usable by the GPU.
- **vLLM has no mature Metal/MPS backend.** Locally it runs CPU-only. The
  vLLM adapter treats every GPU-specific metric (KV-cache %, device) as
  optional and defaults `BackendDevice="cpu"` rather than erroring when
  they're absent -- this is exercised directly by
  `TestVLLMMissingGPUMetricsIsGraceful`. Because of this, the numbers a real
  local vLLM instance would produce here are CPU-mode numbers, not
  GPU-mode ones; end-to-end verification in this repo instead uses a small
  `fakevllm` test server ([`test/fakevllm`](test/fakevllm/main.go)) that
  serves realistic vLLM-shaped Prometheus output, so the adapter's parsing
  and concurrency logic is proven against a real HTTP server without
  requiring a GPU box. **GPU-mode behavior (real vLLM on CUDA) has not been
  validated in this session** and would need a rented cloud GPU instance to
  confirm the adapter's assumptions (metric names, device labeling) hold
  against a real accelerated vLLM deployment.

## How to run it

### Prerequisites

- Go 1.26+
- Python 3.9+
- [Ollama](https://ollama.com) running locally, with at least one model
  pulled (`ollama pull qwen2.5:0.5b` is small and fast for trying this out)
- vLLM is optional; everything works with Ollama alone

### Build and run the collector

```bash
make build                       # -> bin/llm-obs-collector, bin/llm-obs
./bin/llm-obs-collector          # polls Ollama + vLLM, serves :8765, stores llm-obs.db
```

### Install and use the Python client

```bash
pip install -e client/
```

```python
from llm_obs import trace

with trace(name="summarize_endpoint", model="qwen2.5:0.5b") as span:
    resp = ollama.generate(model="qwen2.5:0.5b", prompt=prompt)
    span.record_ollama_response(resp)
```

Or try the included example, which fires 50 traced calls at a real local
model and prints latency percentiles:

```bash
python examples/simulate_calls.py --model qwen2.5:0.5b --n 50 --tag version=v1
```

### Use the CLI

```bash
./bin/llm-obs status                                          # runtimes, loaded models, live snapshot
./bin/llm-obs traces --name summarize_endpoint                # latency percentiles
./bin/llm-obs diagnose --slowest --name summarize_endpoint     # plain-language "why was this slow"
./bin/llm-obs compare --before version=v1 --after version=v2   # A/B regression report
./bin/llm-obs check --model llama3.2:8b --context 8192         # will this fit before I load it?
./bin/llm-obs events --since 1h                                # runtime/model state changes
```

### Optional: Prometheus + Grafana (Phase 8, secondary)

```bash
docker compose up --build
open http://localhost:3000    # Grafana, anonymous viewer access, dashboard pre-provisioned
open http://localhost:9090    # Prometheus
```

Ollama/vLLM are intentionally not part of the compose file -- they're your
own local setup; the collector container reaches them via
`host.docker.internal`. **Note:** the compose file and Dockerfile are
syntax-validated (`docker compose config`) but were not build/run-verified
in this session -- the local Docker Desktop daemon needed an administrator
password prompt to finish starting privileged port mapping, which was out
of scope to supply here. The CI workflow (`.github/workflows/ci.yml`) does
build the image on every push, on a runner where the daemon is already
running, so that gap is closed there.

### Run the tests

```bash
make test          # go vet + go test -race for the collector, pytest for the client
```

78 Go tests across 8 packages plus 8 Python tests, all passing as of the
last commit -- see [RESULTS.md](RESULTS.md) for the load and chaos test
results, which exercise real running processes rather than only mocks.

## Repository layout

```
llm-obs/
  collector/                       Go module: the collector daemon + CLI
    cmd/collector/                 daemon entrypoint (main.go)
    cmd/llm-obs/                   CLI entrypoint
    internal/adapters/             Ollama (/api/ps, /api/show) and vLLM (/metrics) adapters
    internal/core/                 normalized schema, ring buffers, polling collector
    internal/core/diagnose/        causal diagnoser (Phase 5) + A/B comparer (Phase 6)
    internal/core/preflight/       pre-flight capacity check (Phase 7)
    internal/trace/                trace ingestion, context-joining, percentile aggregation
    internal/storage/              SQLite persistence
    internal/api/                  the local HTTP API
    internal/export/               Prometheus exporter (Phase 8)
    internal/hostmem/              cross-platform host memory probe
  client/                          Python package: llm_obs (pip install -e client/)
  cli/                             thin wrapper script for running the CLI from source
  examples/simulate_calls.py       fires N traced calls at a real Ollama model
  test/fakevllm/                   a real HTTP server serving vLLM-shaped /metrics for testing
  test/load/                       load-test driver used for Phase 9 (see RESULTS.md)
  deploy/                          Dockerfile, Prometheus config, Grafana provisioning
  docker-compose.yml               optional production-integration stack
  .github/workflows/ci.yml         Go build/vet/test, Python tests, Docker validation
  RESULTS.md                       load & chaos test results against real processes
```

## What's next

- **Validate GPU-mode vLLM.** Everything here about vLLM has been proven
  against its CPU-mode behavior and a faithful fake server; a rented cloud
  GPU box is the natural next step to confirm the adapter's metric-name and
  device-labeling assumptions against a real accelerated deployment.
- **More runtimes.** llama.cpp's server mode and LM Studio's local API are
  natural next adapters given the same `Adapter` interface.
- **OpenTelemetry export.** The trace model already carries most of what an
  OTel span needs (name, timestamps, tags, token counts); exporting spans
  directly would let this plug into an existing tracing backend instead of
  requiring its own SQLite store for teams that already have one.
- **Hosted version.** The collector's HTTP API and SQLite storage are
  already decoupled from any particular machine; a small multi-tenant
  wrapper around the same core would let a team point several developers'
  local runtimes at one shared dashboard.
