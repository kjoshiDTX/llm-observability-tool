// Package trace defines request-level trace records and the ingester that
// correlates each trace with what the runtime was doing at that moment.
package trace

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
)

// Trace is one traced LLM call as reported by the client SDK, enriched by
// the collector with the system context observed during its window.
type Trace struct {
	ID      string            `json:"id"`
	Name    string            `json:"name"`
	Model   string            `json:"model"`
	Runtime string            `json:"runtime"`
	Tags    map[string]string `json:"tags,omitempty"`

	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	DurationMS float64   `json:"duration_ms"`
	TTFTMS     float64   `json:"ttft_ms,omitempty"`

	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSec     float64 `json:"tokens_per_sec"`

	Error string `json:"error,omitempty"`

	// Context is filled by the collector at ingest time.
	Context Context `json:"context"`

	CreatedAt time.Time `json:"created_at"`
}

// Context is the system state joined against a trace.
type Context struct {
	SnapshotCount    int      `json:"snapshot_count"`
	MemUsedMB        int64    `json:"mem_used_mb"`
	MemTotalMB       int64    `json:"mem_total_mb"`
	MemPressure      float64  `json:"mem_pressure"` // 0..1
	KVCacheMaxPct    float64  `json:"kv_cache_max_pct"`
	ContextUsedTok   int      `json:"context_used_tok"`
	ContextMaxTok    int      `json:"context_max_tok"`
	ContextPressure  float64  `json:"context_pressure"` // 0..1
	BackendDevice    string   `json:"backend_device"`   // device at end of window
	BackendDevices   []string `json:"backend_devices"`  // distinct devices seen
	RequestsRunning  int      `json:"requests_running_max"`
	ConcurrentTraces int      `json:"concurrent_traces"`      // other traces on same model overlapping this one
	RuntimeTPS       float64  `json:"runtime_tokens_per_sec"` // mean runtime-reported tps in window (vLLM)
}

// Query filters stored traces.
type Query struct {
	Name  string
	Model string
	Tags  map[string]string
	Since time.Time
	Until time.Time
	Limit int
}

// Store is what the ingester needs from persistence. storage.Store satisfies it.
type Store interface {
	InsertTrace(t Trace) error
	GetTrace(id string) (Trace, bool, error)
	ListTraces(q Query) ([]Trace, error)
	OverlappingTraces(model string, start, end time.Time, excludeID string) ([]Trace, error)
	SnapshotsInRange(runtime, model string, from, to time.Time) ([]core.RuntimeSnapshot, error)
}

// SnapshotSource is what the ingester needs from the live collector.
type SnapshotSource interface {
	History(runtime, model string, from, to time.Time) []core.RuntimeSnapshot
	RuntimeFor(model string) string
	ObserveThroughput(runtime, model string, tps float64, ctxUsed int)
}

// Begin is the lightweight "call started" notification used to count
// concurrency for calls that are still in flight.
type Begin struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Model     string    `json:"model"`
	Runtime   string    `json:"runtime"`
	StartTime time.Time `json:"start_time"`
}

// Ingester validates, enriches and stores traces.
type Ingester struct {
	store Store
	src   SnapshotSource
	Pad   time.Duration // window padding so short calls still catch a poll sample

	// OnIngest is called after a trace is stored (metrics export etc).
	OnIngest func(Trace)

	mu       sync.Mutex
	inflight map[string]Begin
}

// NewIngester builds an ingester; src may be nil in tests.
func NewIngester(store Store, src SnapshotSource) *Ingester {
	return &Ingester{store: store, src: src, Pad: 3 * time.Second, inflight: map[string]Begin{}}
}

// BeginTrace registers an in-flight call.
func (in *Ingester) BeginTrace(b Begin) error {
	if b.ID == "" {
		return errors.New("begin: id required")
	}
	if b.StartTime.IsZero() {
		b.StartTime = time.Now()
	}
	in.mu.Lock()
	in.inflight[b.ID] = b
	// Drop stale in-flight entries (client crashed before completing).
	for id, x := range in.inflight {
		if time.Since(x.StartTime) > time.Hour {
			delete(in.inflight, id)
		}
	}
	in.mu.Unlock()
	return nil
}

// InFlight returns the number of in-flight calls (all models).
func (in *Ingester) InFlight() int {
	in.mu.Lock()
	defer in.mu.Unlock()
	return len(in.inflight)
}

// Ingest validates a client-submitted trace, joins it with system state, and
// stores it. The returned trace includes the computed context.
func (in *Ingester) Ingest(ctx context.Context, t Trace) (Trace, error) {
	if t.ID == "" {
		return t, errors.New("trace: id required")
	}
	if t.Name == "" {
		return t, errors.New("trace: name required")
	}
	if t.StartTime.IsZero() {
		return t, errors.New("trace: start_time required")
	}
	if t.EndTime.IsZero() {
		t.EndTime = time.Now()
	}
	if t.EndTime.Before(t.StartTime) {
		return t, errors.New("trace: end_time before start_time")
	}
	if t.DurationMS <= 0 {
		t.DurationMS = float64(t.EndTime.Sub(t.StartTime)) / float64(time.Millisecond)
	}
	if t.Model == "" {
		t.Model = "unknown"
	}
	if t.Runtime == "" && in.src != nil {
		t.Runtime = in.src.RuntimeFor(t.Model)
	}
	if t.Runtime == "" {
		t.Runtime = "unknown"
	}
	if t.TokensPerSec <= 0 && t.CompletionTokens > 0 {
		gen := t.DurationMS - t.TTFTMS
		if gen <= 0 {
			gen = t.DurationMS
		}
		if gen > 0 {
			t.TokensPerSec = float64(t.CompletionTokens) / (gen / 1000)
		}
	}
	t.CreatedAt = time.Now()

	// Join against snapshots for the window (live ring first, then storage).
	from, to := t.StartTime.Add(-in.Pad), t.EndTime.Add(in.Pad)
	var snaps []core.RuntimeSnapshot
	if in.src != nil {
		snaps = in.src.History(t.Runtime, t.Model, from, to)
	}
	if len(snaps) == 0 && in.store != nil {
		snaps, _ = in.store.SnapshotsInRange(t.Runtime, t.Model, from, to)
	}
	t.Context = BuildContext(t, snaps)

	// Concurrency: in-flight calls plus completed traces that overlap.
	t.Context.ConcurrentTraces = in.concurrent(t)

	in.mu.Lock()
	delete(in.inflight, t.ID)
	in.mu.Unlock()

	if in.src != nil && t.Error == "" {
		in.src.ObserveThroughput(t.Runtime, t.Model, t.TokensPerSec, t.PromptTokens+t.CompletionTokens)
	}
	if in.store != nil {
		if err := in.store.InsertTrace(t); err != nil {
			return t, fmt.Errorf("store trace: %w", err)
		}
	}
	if in.OnIngest != nil {
		in.OnIngest(t)
	}
	return t, nil
}

func (in *Ingester) concurrent(t Trace) int {
	seen := map[string]bool{}
	in.mu.Lock()
	for id, b := range in.inflight {
		if id == t.ID || b.Model != t.Model {
			continue
		}
		if !b.StartTime.After(t.EndTime) {
			seen[id] = true
		}
	}
	in.mu.Unlock()
	if in.store != nil {
		if others, err := in.store.OverlappingTraces(t.Model, t.StartTime, t.EndTime, t.ID); err == nil {
			for _, o := range others {
				seen[o.ID] = true
			}
		}
	}
	return len(seen)
}

// BuildContext summarises a window of snapshots for one trace.
func BuildContext(t Trace, snaps []core.RuntimeSnapshot) Context {
	c := Context{SnapshotCount: len(snaps)}
	c.ContextUsedTok = t.PromptTokens + t.CompletionTokens
	devs := map[string]bool{}
	var tpsSum float64
	var tpsN int
	for _, s := range snaps {
		if s.KVCacheUsedPct > c.KVCacheMaxPct {
			c.KVCacheMaxPct = s.KVCacheUsedPct
		}
		if s.RequestsRunning > c.RequestsRunning {
			c.RequestsRunning = s.RequestsRunning
		}
		if s.ContextMaxTok > 0 {
			c.ContextMaxTok = s.ContextMaxTok
		}
		if s.BackendDevice != "" && s.BackendDevice != "unknown" {
			if !devs[s.BackendDevice] {
				devs[s.BackendDevice] = true
				c.BackendDevices = append(c.BackendDevices, s.BackendDevice)
			}
			c.BackendDevice = s.BackendDevice
		}
		if s.TPSSource == "runtime" && s.TokensPerSec > 0 {
			tpsSum += s.TokensPerSec
			tpsN++
		}
		c.MemUsedMB, c.MemTotalMB = s.MemUsedMB, s.MemTotalMB
		if p := s.MemPressure(); p > c.MemPressure {
			c.MemPressure = p
		}
	}
	if tpsN > 0 {
		c.RuntimeTPS = tpsSum / float64(tpsN)
	}
	if c.ContextMaxTok > 0 {
		c.ContextPressure = float64(c.ContextUsedTok) / float64(c.ContextMaxTok)
	}
	return c
}

// Stats are latency/throughput percentiles for one trace name.
type Stats struct {
	Name      string   `json:"name"`
	Count     int      `json:"count"`
	Errors    int      `json:"errors"`
	Models    []string `json:"models"`
	P50MS     float64  `json:"p50_ms"`
	P95MS     float64  `json:"p95_ms"`
	P99MS     float64  `json:"p99_ms"`
	MeanMS    float64  `json:"mean_ms"`
	MaxMS     float64  `json:"max_ms"`
	TPSP50    float64  `json:"tokens_per_sec_p50"`
	TPSMean   float64  `json:"tokens_per_sec_mean"`
	TTFTP50MS float64  `json:"ttft_p50_ms,omitempty"`
}

// Percentile returns the nearest-rank percentile (p in 0..100) of values.
func Percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	rank := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}

// Aggregate groups traces by name and computes stats, sorted by name.
func Aggregate(traces []Trace) []Stats {
	groups := map[string][]Trace{}
	for _, t := range traces {
		groups[t.Name] = append(groups[t.Name], t)
	}
	names := make([]string, 0, len(groups))
	for n := range groups {
		names = append(names, n)
	}
	sort.Strings(names)
	out := make([]Stats, 0, len(names))
	for _, n := range names {
		out = append(out, Summarize(n, groups[n]))
	}
	return out
}

// Summarize computes stats for one group of traces.
func Summarize(name string, ts []Trace) Stats {
	s := Stats{Name: name, Count: len(ts)}
	var lat, tps, ttft []float64
	models := map[string]bool{}
	var sum float64
	for _, t := range ts {
		models[t.Model] = true
		if t.Error != "" {
			s.Errors++
		}
		lat = append(lat, t.DurationMS)
		sum += t.DurationMS
		if t.DurationMS > s.MaxMS {
			s.MaxMS = t.DurationMS
		}
		if t.TokensPerSec > 0 {
			tps = append(tps, t.TokensPerSec)
		}
		if t.TTFTMS > 0 {
			ttft = append(ttft, t.TTFTMS)
		}
	}
	for m := range models {
		s.Models = append(s.Models, m)
	}
	sort.Strings(s.Models)
	s.P50MS, s.P95MS, s.P99MS = Percentile(lat, 50), Percentile(lat, 95), Percentile(lat, 99)
	if len(lat) > 0 {
		s.MeanMS = sum / float64(len(lat))
	}
	s.TPSP50 = Percentile(tps, 50)
	if len(tps) > 0 {
		var t float64
		for _, v := range tps {
			t += v
		}
		s.TPSMean = t / float64(len(tps))
	}
	s.TTFTP50MS = Percentile(ttft, 50)
	return s
}

// ParseTagSelector parses "key=value" (or "value", which matches the default
// key) into a single-entry tag map.
func ParseTagSelector(sel, defaultKey string) map[string]string {
	sel = strings.TrimSpace(sel)
	if sel == "" {
		return nil
	}
	if k, v, ok := strings.Cut(sel, "="); ok {
		return map[string]string{k: v}
	}
	return map[string]string{defaultKey: sel}
}
