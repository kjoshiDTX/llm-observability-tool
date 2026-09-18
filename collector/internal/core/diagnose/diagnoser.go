// Package diagnose turns a slow trace plus the system snapshots observed
// during its window into a plain-language explanation of *why* it was slow.
//
// This is the primary user-facing output of llm-obs. Everything else in the
// collector exists to feed it.
package diagnose

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// Cause kinds, ordered roughly by how actionable they are.
const (
	KindContextNearLimit = "context_near_limit"
	KindBackendFallback  = "backend_fallback"
	KindColdStart        = "cold_start"
	KindContention       = "contention"
	KindMemoryPressure   = "memory_pressure"
	KindRegression       = "unexplained_regression"
)

// Cause is one attributed reason for a slowdown.
type Cause struct {
	Kind        string  `json:"kind"`
	Confidence  float64 `json:"confidence"` // 0..1
	Explanation string  `json:"explanation"`
	Suggestion  string  `json:"suggestion,omitempty"`
}

// Baseline summarises this trace's own history (same name + model).
type Baseline struct {
	Count         int     `json:"count"`
	P50MS         float64 `json:"p50_ms"`
	P95MS         float64 `json:"p95_ms"`
	TPSMedian     float64 `json:"tokens_per_sec_median"`
	DurationRatio float64 `json:"duration_ratio"` // this / p50
	TPSRatio      float64 `json:"tps_ratio"`      // this / median
	Device        string  `json:"typical_device"`
}

// TraceRef is a compact pointer to another trace used as evidence.
type TraceRef struct {
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	StartTime  time.Time `json:"start_time"`
	DurationMS float64   `json:"duration_ms"`
}

// Evidence is everything the diagnoser looked at.
type Evidence struct {
	Context          trace.Context          `json:"context"`
	Snapshots        []core.RuntimeSnapshot `json:"snapshots"`
	Events           []core.Event           `json:"events"`
	ConcurrentTraces []TraceRef             `json:"concurrent_traces"`
	Baseline         *Baseline              `json:"baseline,omitempty"`
}

// Diagnosis is the answer to "why was this call slow?".
type Diagnosis struct {
	TraceID    string  `json:"trace_id"`
	Name       string  `json:"name"`
	Model      string  `json:"model"`
	Runtime    string  `json:"runtime"`
	DurationMS float64 `json:"duration_ms"`
	TPS        float64 `json:"tokens_per_sec"`

	// Slow is true when the call was slower than its own baseline (or, with
	// no baseline yet, when a strong cause was detected).
	Slow     bool   `json:"slow"`
	Severity string `json:"severity"` // "ok", "warning", "critical"
	Summary  string `json:"summary"`  // the plain-language headline

	// Causes are the attributed reasons, most confident first. Empty when the
	// call was within its normal range.
	Causes []Cause `json:"causes"`
	// Signals are every condition detected regardless of whether the call was
	// slow (used for A/B pattern comparison and as "notes" on healthy calls).
	Signals  []Cause  `json:"signals"`
	Evidence Evidence `json:"evidence"`
}

// Store is the persistence the diagnoser reads from.
type Store interface {
	GetTrace(id string) (trace.Trace, bool, error)
	SnapshotsInRange(runtime, model string, from, to time.Time) ([]core.RuntimeSnapshot, error)
	OverlappingTraces(model string, start, end time.Time, excludeID string) ([]trace.Trace, error)
	Baseline(name, model string, before time.Time, limit int) ([]trace.Trace, error)
	EventsInRange(from, to time.Time, runtime string) ([]core.Event, error)
}

// Live is the in-memory view (may be nil; storage is the fallback).
type Live interface {
	History(runtime, model string, from, to time.Time) []core.RuntimeSnapshot
}

// Thresholds are the tunable knobs, exported so tests and operators can see
// exactly what "near limit" means.
type Thresholds struct {
	ContextPressure   float64 // fraction of context window used
	KVCachePct        float64 // percent
	MemPressure       float64 // fraction of host memory used
	SlowDurationRatio float64 // this / baseline p50
	SlowTPSRatio      float64 // this / baseline median tps
	MinBaseline       int     // traces needed before baseline comparisons
	ColdStartWindow   time.Duration
	Pad               time.Duration
}

// DefaultThresholds are conservative enough not to fire on ordinary jitter.
var DefaultThresholds = Thresholds{
	ContextPressure:   0.90,
	KVCachePct:        90,
	MemPressure:       0.92,
	SlowDurationRatio: 1.5,
	SlowTPSRatio:      0.7,
	MinBaseline:       5,
	ColdStartWindow:   15 * time.Second,
	Pad:               3 * time.Second,
}

// Diagnoser produces diagnoses.
type Diagnoser struct {
	store Store
	live  Live
	T     Thresholds
}

// New creates a diagnoser with default thresholds.
func New(store Store, live Live) *Diagnoser {
	return &Diagnoser{store: store, live: live, T: DefaultThresholds}
}

// ErrNotFound is returned when the trace id is unknown.
var ErrNotFound = fmt.Errorf("trace not found")

// Diagnose looks up a trace by id and explains it.
func (d *Diagnoser) Diagnose(id string) (any, error) {
	t, ok, err := d.store.GetTrace(id)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, ErrNotFound
	}
	return d.DiagnoseTrace(t), nil
}

// DiagnoseTrace explains an already-loaded trace.
func (d *Diagnoser) DiagnoseTrace(t trace.Trace) Diagnosis {
	ev := d.gather(t)
	return d.Evaluate(t, ev)
}

// gather collects evidence from the live ring and storage.
func (d *Diagnoser) gather(t trace.Trace) Evidence {
	from, to := t.StartTime.Add(-d.T.Pad), t.EndTime.Add(d.T.Pad)
	ev := Evidence{Context: t.Context}
	if d.live != nil {
		ev.Snapshots = d.live.History(t.Runtime, t.Model, from, to)
	}
	if len(ev.Snapshots) == 0 && d.store != nil {
		ev.Snapshots, _ = d.store.SnapshotsInRange(t.Runtime, t.Model, from, to)
	}
	if d.store != nil {
		if evs, err := d.store.EventsInRange(t.StartTime.Add(-d.T.ColdStartWindow), to, t.Runtime); err == nil {
			for _, e := range evs {
				if e.Model == "" || e.Model == t.Model {
					ev.Events = append(ev.Events, e)
				}
			}
		}
		if others, err := d.store.OverlappingTraces(t.Model, t.StartTime, t.EndTime, t.ID); err == nil {
			for _, o := range others {
				ev.ConcurrentTraces = append(ev.ConcurrentTraces, TraceRef{ID: o.ID, Name: o.Name, StartTime: o.StartTime, DurationMS: o.DurationMS})
			}
		}
		if hist, err := d.store.Baseline(t.Name, t.Model, t.StartTime, 50); err == nil && len(hist) >= d.T.MinBaseline {
			ev.Baseline = buildBaseline(t, hist)
		}
	}
	// The stored context already counted in-flight concurrency at ingest;
	// prefer the larger of the two views.
	if len(ev.ConcurrentTraces) > ev.Context.ConcurrentTraces {
		ev.Context.ConcurrentTraces = len(ev.ConcurrentTraces)
	}
	return ev
}

func buildBaseline(t trace.Trace, hist []trace.Trace) *Baseline {
	var lat, tps []float64
	devs := map[string]int{}
	for _, h := range hist {
		lat = append(lat, h.DurationMS)
		if h.TokensPerSec > 0 {
			tps = append(tps, h.TokensPerSec)
		}
		if h.Context.BackendDevice != "" {
			devs[h.Context.BackendDevice]++
		}
	}
	b := &Baseline{Count: len(hist), P50MS: trace.Percentile(lat, 50), P95MS: trace.Percentile(lat, 95), TPSMedian: trace.Percentile(tps, 50)}
	if b.P50MS > 0 {
		b.DurationRatio = t.DurationMS / b.P50MS
	}
	if b.TPSMedian > 0 && t.TokensPerSec > 0 {
		b.TPSRatio = t.TokensPerSec / b.TPSMedian
	}
	best := 0
	for dev, n := range devs {
		if n > best {
			best, b.Device = n, dev
		}
	}
	return b
}

// Evaluate is the pure rule engine: given a trace and its evidence, produce
// the diagnosis. It has no I/O so tests can feed synthetic sequences.
func (d *Diagnoser) Evaluate(t trace.Trace, ev Evidence) Diagnosis {
	out := Diagnosis{TraceID: t.ID, Name: t.Name, Model: t.Model, Runtime: t.Runtime, DurationMS: t.DurationMS, TPS: t.TokensPerSec, Evidence: ev}
	ctx := ev.Context
	T := d.T

	// ---- signals: every condition we can see, independent of slowness ----
	var signals []Cause

	// 1. Context / KV cache near limit.
	if ctx.ContextMaxTok > 0 && ctx.ContextPressure >= T.ContextPressure {
		signals = append(signals, Cause{
			Kind: KindContextNearLimit, Confidence: 0.9,
			Explanation: fmt.Sprintf("Context usage hit %.0f%% of the %d-token window (%d prompt + %d completion tokens). Near the limit the runtime shifts or evicts KV-cache entries and re-prefills, which stalls generation.",
				ctx.ContextPressure*100, ctx.ContextMaxTok, t.PromptTokens, t.CompletionTokens),
			Suggestion: "Trim the prompt, summarise earlier turns, or raise num_ctx (check the pre-flight `llm-obs check` first).",
		})
	} else if ctx.KVCacheMaxPct >= T.KVCachePct {
		signals = append(signals, Cause{
			Kind: KindContextNearLimit, Confidence: 0.85,
			Explanation: fmt.Sprintf("Runtime KV-cache was %.0f%% full during this call. vLLM preempts and recomputes requests when the cache is exhausted, so throughput collapses under this condition.", ctx.KVCacheMaxPct),
			Suggestion:  "Lower max concurrent sequences, shorten max_model_len, or raise gpu_memory_utilization.",
		})
	}

	// 2. Backend fallback (device changed mid-call, or running off-accelerator
	//    when history says it normally runs on one).
	devs := ctx.BackendDevices
	if len(devs) == 0 {
		devs = devicesFromSnapshots(ev.Snapshots)
	}
	changedMidCall := len(devs) > 1
	backendEvent := false
	for _, e := range ev.Events {
		if e.Kind == core.EventBackendChanged && !e.Timestamp.Before(t.StartTime.Add(-T.Pad)) {
			backendEvent = true
		}
	}
	memPct := ctx.MemPressure * 100
	switch {
	case changedMidCall || backendEvent:
		signals = append(signals, Cause{
			Kind: KindBackendFallback, Confidence: 0.95,
			Explanation: fmt.Sprintf("Backend changed during this request (%s) with host memory at %.0f%%: the runtime fell back from the accelerator to CPU mid-call. CPU decode is typically 5-10x slower per token.",
				strings.Join(devs, " -> "), memPct),
			Suggestion: "Free memory (unload other models, close apps) or use a smaller quantization so the model stays fully resident on the GPU.",
		})
	case (ctx.BackendDevice == "cpu" || ctx.BackendDevice == "mixed") && ev.Baseline != nil && ev.Baseline.Device != "" && ev.Baseline.Device != ctx.BackendDevice && ev.Baseline.Device != "cpu":
		signals = append(signals, Cause{
			Kind: KindBackendFallback, Confidence: 0.8,
			Explanation: fmt.Sprintf("This call ran on %q but the last %d calls to this endpoint ran on %q. Part or all of the model was offloaded to CPU (host memory at %.0f%%).",
				ctx.BackendDevice, ev.Baseline.Count, ev.Baseline.Device, memPct),
			Suggestion: "Check `llm-obs status` for other resident models; free memory so the model reloads on the accelerator.",
		})
	}

	// 3. Cold start: model (re)loaded just before or during the call.
	for _, e := range ev.Events {
		if e.Kind == core.EventModelLoaded && e.Model == t.Model && e.Timestamp.After(t.StartTime.Add(-T.ColdStartWindow)) && !e.Timestamp.After(t.EndTime.Add(T.Pad)) {
			signals = append(signals, Cause{
				Kind: KindColdStart, Confidence: 0.85,
				Explanation: fmt.Sprintf("The model was loaded %s before this call started, so this request paid the load + warm-up cost (weights read from disk, graph compiled, KV cache allocated).",
					humanDur(t.StartTime.Sub(e.Timestamp))),
				Suggestion: "Keep the model resident (Ollama: keep_alive=-1) or send a warm-up request after load.",
			})
			break
		}
	}

	// 4. Contention.
	if n := ctx.ConcurrentTraces; n >= 1 {
		conf := 0.55 + 0.1*float64(n)
		if conf > 0.9 {
			conf = 0.9
		}
		expl := fmt.Sprintf("%d other traced request(s) were running against %s at the same time. Local runtimes serialise or batch concurrent requests (Ollama default OLLAMA_NUM_PARALLEL=1), so each call waits on the others.", n, t.Model)
		if ctx.RequestsRunning > 1 {
			expl += fmt.Sprintf(" The runtime itself reported up to %d requests running.", ctx.RequestsRunning)
		}
		signals = append(signals, Cause{Kind: KindContention, Confidence: conf, Explanation: expl,
			Suggestion: "Queue calls client-side, raise the runtime's parallelism if memory allows, or split traffic across models."})
	} else if ctx.RequestsRunning > 1 {
		signals = append(signals, Cause{Kind: KindContention, Confidence: 0.6,
			Explanation: fmt.Sprintf("The runtime reported %d requests running concurrently during this call (untraced callers are sharing the model).", ctx.RequestsRunning),
			Suggestion:  "Trace all callers, or isolate this workload on its own runtime instance."})
	}

	// 5. Memory pressure on its own (no fallback observed yet).
	if ctx.MemPressure >= T.MemPressure && !hasKind(signals, KindBackendFallback) {
		expl := fmt.Sprintf("Host memory was %.0f%% used during this call.", memPct)
		if ctx.MemTotalMB > 0 && ev.Snapshots != nil && len(ev.Snapshots) > 0 && ev.Snapshots[0].UnifiedMemory {
			expl += " On unified memory the GPU competes for the same pool, so the OS compresses or swaps pages that the model needs."
		}
		signals = append(signals, Cause{Kind: KindMemoryPressure, Confidence: 0.6, Explanation: expl,
			Suggestion: "Close other memory-heavy processes or use a smaller quantization; a backend fallback is likely next."})
	}

	sort.SliceStable(signals, func(i, j int) bool { return signals[i].Confidence > signals[j].Confidence })
	out.Signals = signals

	// ---- slowness vs baseline ----
	b := ev.Baseline
	slowByDuration := b != nil && b.DurationRatio >= T.SlowDurationRatio
	slowByTPS := b != nil && b.TPSRatio > 0 && b.TPSRatio <= T.SlowTPSRatio
	strong := false
	for _, s := range signals {
		if s.Confidence >= 0.8 {
			strong = true
		}
	}

	switch {
	case b == nil:
		// No baseline yet: only strong signals are worth reporting as causes.
		if strong {
			out.Slow = true
			out.Causes = filterConf(signals, 0.8)
		}
	case slowByDuration || slowByTPS:
		out.Slow = true
		out.Causes = signals
		if len(out.Causes) == 0 {
			out.Causes = []Cause{{
				Kind: KindRegression, Confidence: 0.5,
				Explanation: fmt.Sprintf("This call took %.0f ms (%.1fx its %.0f ms baseline p50) at %.1f tok/s (baseline median %.1f), but no contention, context, memory or backend signal explains it. Flagging as an unexplained regression.",
					t.DurationMS, b.DurationRatio, b.P50MS, t.TokensPerSec, b.TPSMedian),
				Suggestion: "Check for a model/runtime version change, thermal throttling, or background CPU load; compare with `llm-obs compare` across versions.",
			}}
		}
	}

	// ---- headline ----
	switch {
	case out.Slow && len(out.Causes) > 0:
		top := out.Causes[0]
		out.Severity = "warning"
		if top.Confidence >= 0.85 {
			out.Severity = "critical"
		}
		prefix := fmt.Sprintf("Slow: %.0f ms", t.DurationMS)
		if b != nil {
			prefix += fmt.Sprintf(" (%.1fx baseline p50 of %.0f ms)", b.DurationRatio, b.P50MS)
		}
		out.Summary = prefix + " - " + kindLabel(top.Kind) + ". " + top.Explanation
	case b != nil:
		out.Severity = "ok"
		out.Summary = fmt.Sprintf("Within normal range: %.0f ms vs baseline p50 %.0f ms (p95 %.0f ms) over %d prior calls.", t.DurationMS, b.P50MS, b.P95MS, b.Count)
		if len(signals) > 0 {
			out.Summary += " Note: " + signals[0].Explanation
		}
	default:
		out.Severity = "ok"
		out.Summary = fmt.Sprintf("No anomaly detected in %.0f ms call; fewer than %d prior calls to %q on %s, so no baseline comparison yet.", t.DurationMS, T.MinBaseline, t.Name, t.Model)
	}
	if out.Causes == nil {
		out.Causes = []Cause{}
	}
	if out.Signals == nil {
		out.Signals = []Cause{}
	}
	return out
}

func devicesFromSnapshots(snaps []core.RuntimeSnapshot) []string {
	var out []string
	seen := map[string]bool{}
	for _, s := range snaps {
		if s.BackendDevice != "" && s.BackendDevice != "unknown" && !seen[s.BackendDevice] {
			seen[s.BackendDevice] = true
			out = append(out, s.BackendDevice)
		}
	}
	return out
}

func hasKind(cs []Cause, kind string) bool {
	for _, c := range cs {
		if c.Kind == kind {
			return true
		}
	}
	return false
}

func filterConf(cs []Cause, min float64) []Cause {
	var out []Cause
	for _, c := range cs {
		if c.Confidence >= min {
			out = append(out, c)
		}
	}
	return out
}

func kindLabel(kind string) string {
	switch kind {
	case KindContextNearLimit:
		return "context near limit, likely evicting cache"
	case KindBackendFallback:
		return "GPU->CPU fallback during this request"
	case KindColdStart:
		return "cold start (model just loaded)"
	case KindContention:
		return "contention: concurrent requests to same model"
	case KindMemoryPressure:
		return "host memory pressure"
	case KindRegression:
		return "unexplained regression vs baseline, flagging for investigation"
	}
	return kind
}

func humanDur(d time.Duration) string {
	if d < 0 {
		return "0s"
	}
	if d < time.Second {
		return fmt.Sprintf("%d ms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1f s", d.Seconds())
}
