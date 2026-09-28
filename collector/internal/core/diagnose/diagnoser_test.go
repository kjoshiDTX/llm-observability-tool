package diagnose

import (
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// baseTrace returns a trace that, on its own (no evidence), is unremarkable:
// no signals should fire against it unless a test deliberately adds one.
func baseTrace() trace.Trace {
	return trace.Trace{
		ID: "t1", Name: "summarize", Model: "m", Runtime: "ollama",
		StartTime: time.Now(), EndTime: time.Now().Add(500 * time.Millisecond),
		DurationMS: 500, CompletionTokens: 50, TokensPerSec: 100,
	}
}

func healthyBaseline() *Baseline {
	return &Baseline{Count: 20, P50MS: 480, P95MS: 600, TPSMedian: 100, DurationRatio: 500.0 / 480.0, TPSRatio: 1.0, Device: "mps"}
}

func slowBaseline(durationRatio, tpsRatio float64) *Baseline {
	return &Baseline{Count: 20, P50MS: 480, P95MS: 600, TPSMedian: 100, DurationRatio: durationRatio, TPSRatio: tpsRatio, Device: "mps"}
}

func newDiagnoser() *Diagnoser { return New(nil, nil) }

// ---- Cause 1: context / KV cache near limit ----

func TestDiagnosis_ContextNearLimit(t *testing.T) {
	tr := baseTrace()
	tr.PromptTokens, tr.CompletionTokens = 3800, 100
	ev := Evidence{
		Context:  trace.Context{ContextUsedTok: 3900, ContextMaxTok: 4096, ContextPressure: 3900.0 / 4096, BackendDevice: "mps"},
		Baseline: slowBaseline(1.6, 0.6),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow {
		t.Fatal("expected slow=true")
	}
	if len(d.Causes) == 0 || d.Causes[0].Kind != KindContextNearLimit {
		t.Fatalf("top cause = %+v, want %s", d.Causes, KindContextNearLimit)
	}
	if d.Severity != "critical" {
		t.Errorf("severity = %s, want critical (confidence 0.9)", d.Severity)
	}
}

func TestDiagnosis_KVCacheNearLimit_NoContextMax(t *testing.T) {
	// vLLM-style: no context window info, but KV cache reported directly.
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{KVCacheMaxPct: 95, BackendDevice: "cuda"},
		Baseline: slowBaseline(1.7, 0.5),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindContextNearLimit {
		t.Fatalf("expected context_near_limit cause, got %+v", d)
	}
}

func TestDiagnosis_ContextBelowThreshold_NoFalsePositive(t *testing.T) {
	// 80% context usage is below the 90% threshold: must not fire.
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{ContextUsedTok: 3277, ContextMaxTok: 4096, ContextPressure: 0.80, BackendDevice: "mps"},
		Baseline: healthyBaseline(),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindContextNearLimit) {
		t.Errorf("context signal fired below threshold: %+v", d.Signals)
	}
	if d.Slow {
		t.Errorf("healthy call marked slow: %+v", d)
	}
}

// ---- Cause 2: backend fallback ----

func TestDiagnosis_BackendFallbackMidCall(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{BackendDevice: "cpu", BackendDevices: []string{"mps", "cpu"}, MemPressure: 0.85},
		Snapshots: []core.RuntimeSnapshot{
			{BackendDevice: "mps", Timestamp: tr.StartTime},
			{BackendDevice: "cpu", Timestamp: tr.StartTime.Add(200 * time.Millisecond)},
		},
		Baseline: slowBaseline(2.0, 0.4),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindBackendFallback {
		t.Fatalf("expected backend_fallback, got %+v", d.Causes)
	}
	if d.Severity != "critical" {
		t.Errorf("severity = %s, want critical", d.Severity)
	}
}

func TestDiagnosis_BackendFallbackViaBackendChangedEvent(t *testing.T) {
	// Only one device seen in the trace's own context, but a backend_changed
	// event landed inside the call's window (e.g. observed by a concurrent poll
	// this trace's own snapshots window didn't capture).
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{BackendDevice: "cpu", MemPressure: 0.8},
		Events: []core.Event{
			{Kind: core.EventBackendChanged, Model: "m", Timestamp: tr.StartTime.Add(100 * time.Millisecond)},
		},
		Baseline: slowBaseline(1.8, 0.5),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindBackendFallback {
		t.Fatalf("expected backend_fallback via event, got %+v", d.Causes)
	}
}

func TestDiagnosis_BackendFallbackVsHistoricalBaseline(t *testing.T) {
	// This call ran entirely on CPU; no device change observed within the call,
	// but the last N calls to this endpoint ran on mps. Lower confidence (0.8)
	// than an observed mid-call change (0.95).
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{BackendDevice: "cpu", MemPressure: 0.7},
		Baseline: &Baseline{Count: 10, P50MS: 480, P95MS: 600, TPSMedian: 100, DurationRatio: 1.6, TPSRatio: 0.55, Device: "mps"},
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindBackendFallback {
		t.Fatalf("expected backend_fallback vs baseline, got %+v", d.Causes)
	}
	if d.Causes[0].Confidence != 0.8 {
		t.Errorf("confidence = %v, want 0.8 for baseline-inferred fallback", d.Causes[0].Confidence)
	}
}

func TestDiagnosis_NoFallbackWhenBaselineDeviceIsAlsoCPU(t *testing.T) {
	// vLLM on Apple Silicon runs CPU-only always: cpu-vs-cpu must not be
	// reported as a fallback.
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{BackendDevice: "cpu", MemPressure: 0.5},
		Baseline: &Baseline{Count: 10, P50MS: 480, P95MS: 600, TPSMedian: 100, DurationRatio: 1.6, TPSRatio: 0.55, Device: "cpu"},
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindBackendFallback) {
		t.Errorf("cpu-vs-cpu baseline should never be a fallback: %+v", d.Signals)
	}
}

// ---- Cause 3: cold start ----

func TestDiagnosis_ColdStart(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{BackendDevice: "mps"},
		Events: []core.Event{
			{Kind: core.EventModelLoaded, Model: "m", Timestamp: tr.StartTime.Add(-2 * time.Second)},
		},
		// No baseline yet: this is a strong (0.85) signal, so it should still
		// surface as a cause even without a baseline to compare against.
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow {
		t.Fatalf("cold start with no baseline should still be flagged (strong signal): %+v", d)
	}
	if len(d.Causes) == 0 || d.Causes[0].Kind != KindColdStart {
		t.Fatalf("expected cold_start cause, got %+v", d.Causes)
	}
}

func TestDiagnosis_ColdStart_LoadEventOutsideWindow_DoesNotFire(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{BackendDevice: "mps"},
		Events: []core.Event{
			// Loaded a long time before ColdStartWindow (15s default).
			{Kind: core.EventModelLoaded, Model: "m", Timestamp: tr.StartTime.Add(-time.Minute)},
		},
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindColdStart) {
		t.Errorf("stale load event should not trigger cold_start: %+v", d.Signals)
	}
}

func TestDiagnosis_ColdStart_DifferentModel_DoesNotFire(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Events: []core.Event{
			{Kind: core.EventModelLoaded, Model: "other-model", Timestamp: tr.StartTime.Add(-time.Second)},
		},
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindColdStart) {
		t.Errorf("load event for a different model should not fire: %+v", d.Signals)
	}
}

// ---- Cause 4: contention ----

func TestDiagnosis_Contention(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{ConcurrentTraces: 3, RequestsRunning: 4, BackendDevice: "mps"},
		ConcurrentTraces: []TraceRef{
			{ID: "a", Name: "summarize", DurationMS: 400}, {ID: "b", Name: "summarize", DurationMS: 420}, {ID: "c", Name: "chat", DurationMS: 300},
		},
		Baseline: slowBaseline(1.6, 0.6),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindContention {
		t.Fatalf("expected contention, got %+v", d.Causes)
	}
	// 0.55 + 0.1*3 = 0.85
	if d.Causes[0].Confidence < 0.84 || d.Causes[0].Confidence > 0.86 {
		t.Errorf("confidence = %v, want ~0.85", d.Causes[0].Confidence)
	}
}

func TestDiagnosis_ContentionFromRuntimeOnly(t *testing.T) {
	// No traced concurrent calls, but the runtime itself reports >1 running
	// (untraced callers sharing the model).
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{RequestsRunning: 3, BackendDevice: "cuda"},
		Baseline: slowBaseline(1.6, 0.6),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !hasKind(d.Signals, KindContention) {
		t.Fatalf("expected contention from requests_running alone: %+v", d.Signals)
	}
}

func TestDiagnosis_NoContentionWhenSolo(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{Context: trace.Context{ConcurrentTraces: 0, RequestsRunning: 1, BackendDevice: "mps"}, Baseline: healthyBaseline()}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindContention) {
		t.Errorf("solo call should not show contention: %+v", d.Signals)
	}
}

// ---- Cause 5: memory pressure ----

func TestDiagnosis_MemoryPressureAlone(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context:   trace.Context{MemPressure: 0.95, BackendDevice: "mps"},
		Snapshots: []core.RuntimeSnapshot{{UnifiedMemory: true}},
		Baseline:  slowBaseline(1.6, 0.6),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || !hasKind(d.Causes, KindMemoryPressure) {
		t.Fatalf("expected memory_pressure, got %+v", d.Causes)
	}
}

func TestDiagnosis_MemoryPressureSuppressedWhenFallbackAlreadyExplainsIt(t *testing.T) {
	// When a backend fallback is already detected, memory_pressure is folded
	// into that explanation rather than reported as a second, redundant cause.
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{BackendDevice: "cpu", BackendDevices: []string{"mps", "cpu"}, MemPressure: 0.96},
		Baseline: slowBaseline(2.0, 0.3),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if hasKind(d.Signals, KindMemoryPressure) {
		t.Errorf("memory_pressure should be suppressed once backend_fallback fires: %+v", d.Signals)
	}
	if d.Causes[0].Kind != KindBackendFallback {
		t.Errorf("top cause should be backend_fallback: %+v", d.Causes)
	}
}

// ---- Cause 6: unexplained regression ----

func TestDiagnosis_UnexplainedRegression(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context:  trace.Context{BackendDevice: "mps", MemPressure: 0.4, ConcurrentTraces: 0, RequestsRunning: 1},
		Baseline: slowBaseline(1.8, 0.5),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow {
		t.Fatal("expected slow=true")
	}
	if len(d.Causes) != 1 || d.Causes[0].Kind != KindRegression {
		t.Fatalf("expected a single unexplained_regression cause, got %+v", d.Causes)
	}
	if d.Causes[0].Confidence != 0.5 {
		t.Errorf("regression confidence = %v, want 0.5", d.Causes[0].Confidence)
	}
	if d.Severity != "warning" {
		t.Errorf("severity = %s, want warning (0.5 < 0.85)", d.Severity)
	}
}

// ---- Healthy / no-false-positive cases ----

func TestDiagnosis_HealthyCallWithNoisySnapshots_NoFalsePositive(t *testing.T) {
	// Realistic "noisy but fine" data: some jitter in every metric, all safely
	// under thresholds, plenty of history. Must not diagnose anything.
	tr := baseTrace()
	tr.DurationMS = 510 // within normal range of baseline p50=480
	ev := Evidence{
		Context: trace.Context{
			ContextUsedTok: 1200, ContextMaxTok: 4096, ContextPressure: 1200.0 / 4096,
			KVCacheMaxPct: 40, MemPressure: 0.55, BackendDevice: "mps", RequestsRunning: 1, ConcurrentTraces: 0,
		},
		Snapshots: []core.RuntimeSnapshot{
			{BackendDevice: "mps", TokensPerSec: 98}, {BackendDevice: "mps", TokensPerSec: 101}, {BackendDevice: "mps", TokensPerSec: 100},
		},
		Baseline: healthyBaseline(),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if d.Slow {
		t.Errorf("healthy noisy call incorrectly flagged as slow: %+v", d)
	}
	if len(d.Signals) != 0 {
		t.Errorf("expected zero signals on healthy data, got %+v", d.Signals)
	}
	if d.Severity != "ok" {
		t.Errorf("severity = %s, want ok", d.Severity)
	}
	if len(d.Causes) != 0 {
		t.Errorf("causes should be empty on a healthy call: %+v", d.Causes)
	}
}

func TestDiagnosis_NoBaselineYet_WeakSignalDoesNotMarkSlow(t *testing.T) {
	// A weak-confidence signal (contention with n=1 -> 0.65) should not, on
	// its own with no baseline, be enough to call the request "slow" -- only
	// strong (>=0.8) signals get that benefit before a baseline exists.
	tr := baseTrace()
	ev := Evidence{Context: trace.Context{ConcurrentTraces: 1, BackendDevice: "mps"}}
	d := newDiagnoser().Evaluate(tr, ev)
	if d.Slow {
		t.Errorf("weak signal with no baseline should not mark slow: %+v", d)
	}
	if !hasKind(d.Signals, KindContention) {
		t.Error("contention should still be recorded as a signal even though not 'slow'")
	}
	if d.Severity != "ok" {
		t.Errorf("severity = %s, want ok", d.Severity)
	}
}

func TestDiagnosis_NoBaselineYet_StrongSignalStillFlags(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Events: []core.Event{{Kind: core.EventModelLoaded, Model: "m", Timestamp: tr.StartTime.Add(-time.Second)}},
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if !d.Slow || d.Causes[0].Kind != KindColdStart {
		t.Fatalf("strong signal (cold start) should flag slow even with no baseline: %+v", d)
	}
}

func TestDiagnosis_NoBaselineNoSignals(t *testing.T) {
	tr := baseTrace()
	d := newDiagnoser().Evaluate(tr, Evidence{Context: trace.Context{BackendDevice: "mps"}})
	if d.Slow || d.Severity != "ok" {
		t.Errorf("clean call with no baseline should be ok/not-slow: %+v", d)
	}
}

// ---- Ordering / severity ----

func TestDiagnosis_MultipleCauses_SortedByConfidenceMostConfidentFirst(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{
		Context: trace.Context{
			ContextUsedTok: 3900, ContextMaxTok: 4096, ContextPressure: 3900.0 / 4096, // 0.90
			ConcurrentTraces: 1, // ~0.65
			BackendDevice:    "mps",
		},
		Baseline: slowBaseline(1.8, 0.5),
	}
	d := newDiagnoser().Evaluate(tr, ev)
	if len(d.Causes) < 2 {
		t.Fatalf("expected multiple causes, got %+v", d.Causes)
	}
	for i := 1; i < len(d.Causes); i++ {
		if d.Causes[i].Confidence > d.Causes[i-1].Confidence {
			t.Fatalf("causes not sorted by descending confidence: %+v", d.Causes)
		}
	}
	if d.Causes[0].Kind != KindContextNearLimit {
		t.Errorf("expected context_near_limit (0.9) to rank above contention (~0.65), got %+v", d.Causes[0])
	}
}

func TestDiagnosis_Summary_ReferencesBaselineAndTopCause(t *testing.T) {
	tr := baseTrace()
	ev := Evidence{Context: trace.Context{ConcurrentTraces: 2, BackendDevice: "mps"}, Baseline: slowBaseline(1.7, 0.5)}
	d := newDiagnoser().Evaluate(tr, ev)
	if d.Summary == "" {
		t.Fatal("summary should never be empty")
	}
	if !hasSubstr(d.Summary, "Slow:") || !hasSubstr(d.Summary, "baseline p50") {
		t.Errorf("summary should reference baseline: %q", d.Summary)
	}
}

func TestDiagnosis_HealthySummaryNotesWeakSignal(t *testing.T) {
	tr := baseTrace()
	tr.DurationMS = 490
	ev := Evidence{Context: trace.Context{ConcurrentTraces: 1, BackendDevice: "mps"}, Baseline: healthyBaseline()}
	d := newDiagnoser().Evaluate(tr, ev)
	if d.Slow {
		t.Fatalf("within-baseline duration should not be slow even with a weak signal: %+v", d)
	}
	if !hasSubstr(d.Summary, "Within normal range") || !hasSubstr(d.Summary, "Note:") {
		t.Errorf("healthy-with-signal summary should note it: %q", d.Summary)
	}
}

// ---- Diagnose(id) / DiagnoseTrace wiring, including gather() ----

type fakeDStore struct {
	traces   map[string]trace.Trace
	snaps    []core.RuntimeSnapshot
	events   []core.Event
	overlap  []trace.Trace
	baseline []trace.Trace
}

func (f *fakeDStore) GetTrace(id string) (trace.Trace, bool, error) {
	t, ok := f.traces[id]
	return t, ok, nil
}
func (f *fakeDStore) SnapshotsInRange(rt, model string, from, to time.Time) ([]core.RuntimeSnapshot, error) {
	return f.snaps, nil
}
func (f *fakeDStore) OverlappingTraces(model string, start, end time.Time, exclude string) ([]trace.Trace, error) {
	return f.overlap, nil
}
func (f *fakeDStore) Baseline(name, model string, before time.Time, limit int) ([]trace.Trace, error) {
	return f.baseline, nil
}
func (f *fakeDStore) EventsInRange(from, to time.Time, runtime string) ([]core.Event, error) {
	return f.events, nil
}

func TestDiagnose_UnknownID_ReturnsErrNotFound(t *testing.T) {
	d := New(&fakeDStore{traces: map[string]trace.Trace{}}, nil)
	_, err := d.Diagnose("nope")
	if err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestDiagnose_GatherRespectsMinBaseline(t *testing.T) {
	tr := baseTrace()
	store := &fakeDStore{
		traces:   map[string]trace.Trace{tr.ID: tr},
		baseline: []trace.Trace{{DurationMS: 400}, {DurationMS: 410}}, // only 2, MinBaseline default is 5
	}
	d := New(store, nil)
	got, err := d.Diagnose(tr.ID)
	if err != nil {
		t.Fatal(err)
	}
	diag := got.(Diagnosis)
	if diag.Evidence.Baseline != nil {
		t.Errorf("baseline should be nil below MinBaseline threshold, got %+v", diag.Evidence.Baseline)
	}
}

func TestDiagnose_GatherBuildsBaselineOnceEnoughHistory(t *testing.T) {
	tr := baseTrace()
	tr.DurationMS = 900
	var hist []trace.Trace
	for i := 0; i < 10; i++ {
		hist = append(hist, trace.Trace{DurationMS: 500, TokensPerSec: 100, Context: trace.Context{BackendDevice: "mps"}})
	}
	store := &fakeDStore{traces: map[string]trace.Trace{tr.ID: tr}, baseline: hist}
	d := New(store, nil)
	got, err := d.Diagnose(tr.ID)
	if err != nil {
		t.Fatal(err)
	}
	diag := got.(Diagnosis)
	if diag.Evidence.Baseline == nil {
		t.Fatal("expected baseline to be built with 10 prior traces")
	}
	if diag.Evidence.Baseline.P50MS != 500 {
		t.Errorf("baseline p50 = %v, want 500", diag.Evidence.Baseline.P50MS)
	}
	if !diag.Slow { // 900/500 = 1.8 >= 1.5 threshold
		t.Errorf("900ms vs baseline 500ms p50 should be flagged slow: %+v", diag)
	}
}

func TestDiagnose_GatherMergesOverlappingTracesIntoConcurrency(t *testing.T) {
	tr := baseTrace()
	tr.Context.ConcurrentTraces = 0 // ingest-time view saw none
	store := &fakeDStore{
		traces:  map[string]trace.Trace{tr.ID: tr},
		overlap: []trace.Trace{{ID: "x", Name: "other", DurationMS: 300}}, // but storage now shows one overlapping trace
	}
	d := New(store, nil)
	got, _ := d.Diagnose(tr.ID)
	diag := got.(Diagnosis)
	if diag.Evidence.Context.ConcurrentTraces != 1 {
		t.Errorf("concurrent traces should be backfilled from storage: %+v", diag.Evidence.Context)
	}
}

func hasSubstr(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
