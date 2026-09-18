// Package core holds the normalized runtime schema, the polling collector,
// ring buffers, and the causal diagnoser.
package core

import (
	"context"
	"errors"
	"time"
)

// RuntimeSnapshot is the normalized view of one (runtime, model) pair at one
// instant. Every adapter maps its native format into this struct.
//
// Memory fields are labelled Mem*, not VRAM*, because on Apple Silicon the
// GPU and CPU share one unified pool; UnifiedMemory tells consumers which
// interpretation applies.
type RuntimeSnapshot struct {
	Runtime string `json:"runtime"` // "ollama", "vllm"
	Model   string `json:"model"`

	// Throughput. For vLLM this comes from its /metrics endpoint. Ollama has
	// no metrics endpoint, so the collector derives it from recent traces and
	// sets TPSSource="traces".
	TokensPerSec float64 `json:"tokens_per_sec"`
	TPSSource    string  `json:"tps_source,omitempty"` // "runtime", "traces", ""

	// Host memory (system-wide). Pressure = MemUsedMB / MemTotalMB.
	MemUsedMB     int64 `json:"mem_used_mb"`
	MemTotalMB    int64 `json:"mem_total_mb"`
	UnifiedMemory bool  `json:"unified_memory"`

	// Model residency: how much of this model is resident, and how much of
	// that is on the accelerator. ModelGPUMemMB < ModelMemMB means partial
	// CPU offload.
	ModelMemMB    int64 `json:"model_mem_mb"`
	ModelGPUMemMB int64 `json:"model_gpu_mem_mb"`

	// Cache / context utilisation. Runtimes that do not expose a value leave
	// it at zero and the diagnoser falls back to trace-level token counts.
	KVCacheUsedPct float64 `json:"kv_cache_used_pct"`
	ContextUsedTok int     `json:"context_used_tok"`
	ContextMaxTok  int     `json:"context_max_tok"`

	// Concurrency as seen by the runtime (vLLM exposes running/waiting).
	RequestsRunning int `json:"requests_running"`
	RequestsWaiting int `json:"requests_waiting"`

	// Where the model is executing: "mps", "cuda", "cpu", "mixed".
	BackendDevice string `json:"backend_device"`

	// Descriptive metadata used by the pre-flight check.
	ParameterSize string `json:"parameter_size,omitempty"` // e.g. "8.0B"
	Quantization  string `json:"quantization,omitempty"`   // e.g. "Q4_K_M"

	Timestamp time.Time `json:"timestamp"`
}

// Key identifies the ring buffer a snapshot belongs to.
func (s RuntimeSnapshot) Key() string { return s.Runtime + "/" + s.Model }

// MemPressure returns used/total in [0,1], or 0 if total is unknown.
func (s RuntimeSnapshot) MemPressure() float64 {
	if s.MemTotalMB <= 0 {
		return 0
	}
	return float64(s.MemUsedMB) / float64(s.MemTotalMB)
}

// ContextPressure returns used/max context in [0,1], or 0 if unknown.
func (s RuntimeSnapshot) ContextPressure() float64 {
	if s.ContextMaxTok <= 0 {
		return 0
	}
	return float64(s.ContextUsedTok) / float64(s.ContextMaxTok)
}

// ErrRuntimeUnavailable is returned (wrapped) by adapters when the runtime
// process cannot be reached. It is expected and must never crash the poller.
var ErrRuntimeUnavailable = errors.New("runtime unavailable")

// Adapter is implemented once per supported runtime.
type Adapter interface {
	Name() string
	// Poll returns one snapshot per loaded model. An empty slice with a nil
	// error means "runtime is up, nothing loaded". ErrRuntimeUnavailable means
	// the runtime is down.
	Poll(ctx context.Context) ([]RuntimeSnapshot, error)
}

// EventKind enumerates the state transitions the collector detects.
type EventKind string

const (
	EventRuntimeUp      EventKind = "runtime_up"
	EventRuntimeDown    EventKind = "runtime_down"
	EventModelLoaded    EventKind = "model_loaded"
	EventModelUnloaded  EventKind = "model_unloaded"
	EventBackendChanged EventKind = "backend_changed"
	EventPollError      EventKind = "poll_error"
)

// Event is a discrete state change emitted by the collector.
type Event struct {
	Kind      EventKind `json:"kind"`
	Runtime   string    `json:"runtime"`
	Model     string    `json:"model,omitempty"`
	Detail    string    `json:"detail,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}
