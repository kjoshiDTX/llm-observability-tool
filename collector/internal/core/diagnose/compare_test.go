package diagnose

import (
	"strings"
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

func mkTrace(id string, durMS, tps float64, errStr string, ctx trace.Context) trace.Trace {
	return trace.Trace{ID: id, Name: "summarize", Model: "m", Runtime: "ollama", DurationMS: durMS, TokensPerSec: tps, Error: errStr, Context: ctx}
}

func TestDelta_PctAndDirection(t *testing.T) {
	d := delta("latency_p50_ms", 100, 150, false) // lower is better
	if d.Pct != 50 || d.Better {
		t.Errorf("100->150 latency: pct=%v better=%v, want 50/false", d.Pct, d.Better)
	}
	d = delta("tokens_per_sec_p50", 100, 150, true) // higher is better
	if d.Pct != 50 || !d.Better {
		t.Errorf("100->150 tps: pct=%v better=%v, want 50/true", d.Pct, d.Better)
	}
	d = delta("latency_p50_ms", 0, 0, false)
	if d.Pct != 0 {
		t.Errorf("0->0 should be 0%% change, got %v", d.Pct)
	}
	d = delta("latency_p50_ms", 0, 50, false)
	if d.Pct != 100 {
		t.Errorf("0->50 from zero baseline should read as 100%%, got %v", d.Pct)
	}
}

func TestCompareTraces_LatencyImprovement(t *testing.T) {
	c := NewComparer(nil, nil)
	var before, after []trace.Trace
	for i := 0; i < 30; i++ {
		before = append(before, mkTrace("b", 500, 50, "", trace.Context{BackendDevice: "mps"}))
		after = append(after, mkTrace("a", 300, 80, "", trace.Context{BackendDevice: "mps"}))
	}
	cmp := c.CompareTraces("summarize", map[string]string{"version": "v1"}, map[string]string{"version": "v2"}, before, after)
	if cmp.Before.Count != 30 || cmp.After.Count != 30 {
		t.Fatalf("counts: %+v %+v", cmp.Before, cmp.After)
	}
	p50 := cmp.Deltas[0]
	if p50.Before != 500 || p50.After != 300 || !p50.Better {
		t.Errorf("p50 delta: %+v", p50)
	}
	if !strings.Contains(cmp.Verdict, "faster") {
		t.Errorf("verdict should mention faster: %q", cmp.Verdict)
	}
	if !strings.Contains(cmp.Verdict, "throughput up") {
		t.Errorf("verdict should mention throughput improvement: %q", cmp.Verdict)
	}
}

func TestCompareTraces_LatencyRegression(t *testing.T) {
	c := NewComparer(nil, nil)
	var before, after []trace.Trace
	for i := 0; i < 20; i++ {
		before = append(before, mkTrace("b", 300, 90, "", trace.Context{BackendDevice: "mps"}))
		after = append(after, mkTrace("a", 600, 40, "", trace.Context{BackendDevice: "mps"}))
	}
	cmp := c.CompareTraces("", nil, nil, before, after)
	if !strings.Contains(cmp.Verdict, "slower") {
		t.Errorf("verdict should mention slower: %q", cmp.Verdict)
	}
	if !strings.Contains(cmp.Verdict, "throughput down") {
		t.Errorf("verdict should mention throughput drop: %q", cmp.Verdict)
	}
}

func TestCompareTraces_RoughlyUnchanged(t *testing.T) {
	c := NewComparer(nil, nil)
	var before, after []trace.Trace
	for i := 0; i < 20; i++ {
		before = append(before, mkTrace("b", 500, 50, "", trace.Context{}))
		after = append(after, mkTrace("a", 520, 49, "", trace.Context{}))
	}
	cmp := c.CompareTraces("", nil, nil, before, after)
	if !strings.Contains(cmp.Verdict, "roughly unchanged") {
		t.Errorf("verdict should say roughly unchanged for a <10%% delta: %q", cmp.Verdict)
	}
}

func TestCompareTraces_ErrorRateIncrease(t *testing.T) {
	c := NewComparer(nil, nil)
	var before, after []trace.Trace
	for i := 0; i < 20; i++ {
		before = append(before, mkTrace("b", 500, 50, "", trace.Context{}))
	}
	for i := 0; i < 20; i++ {
		errStr := ""
		if i < 4 {
			errStr = "timeout"
		}
		after = append(after, mkTrace("a", 500, 50, errStr, trace.Context{}))
	}
	cmp := c.CompareTraces("", nil, nil, before, after)
	if cmp.After.Errors != 4 || cmp.After.ErrorRate != 0.2 {
		t.Errorf("after error stats: %+v", cmp.After)
	}
	if !strings.Contains(cmp.Verdict, "error rate rose") {
		t.Errorf("verdict should mention rising error rate: %q", cmp.Verdict)
	}
}

// This is the scenario named explicitly in the plan: "fallback events appeared
// in 12% of 'after' runs vs 0% in 'before'". Build that exact shape and check
// the pattern-delta output and verdict both surface it.
func TestCompareTraces_FallbackPatternAppearsInAfterOnly(t *testing.T) {
	diag := New(nil, nil)
	c := NewComparer(nil, diag)

	var before []trace.Trace
	for i := 0; i < 100; i++ {
		before = append(before, mkTrace("b", 400, 100, "", trace.Context{BackendDevice: "mps"}))
	}
	var after []trace.Trace
	for i := 0; i < 100; i++ {
		ctx := trace.Context{BackendDevice: "mps"}
		if i < 12 {
			// Backend fell back to CPU mid-call for 12% of 'after' runs.
			ctx = trace.Context{BackendDevice: "cpu", BackendDevices: []string{"mps", "cpu"}, MemPressure: 0.9}
		}
		after = append(after, mkTrace("a", 400, 100, "", ctx))
	}
	cmp := c.CompareTraces("summarize", map[string]string{"version": "v1"}, map[string]string{"version": "v2"}, before, after)

	var fb *PatternDelta
	for i := range cmp.Patterns {
		if cmp.Patterns[i].Kind == KindBackendFallback {
			fb = &cmp.Patterns[i]
		}
	}
	if fb == nil {
		t.Fatal("expected a backend_fallback pattern entry")
	}
	if fb.BeforeCount != 0 || fb.AfterCount != 12 {
		t.Errorf("fallback counts: before=%d after=%d, want 0/12", fb.BeforeCount, fb.AfterCount)
	}
	if fb.BeforePct != 0 || fb.AfterPct != 12 {
		t.Errorf("fallback pct: before=%v after=%v, want 0/12", fb.BeforePct, fb.AfterPct)
	}
	if !strings.Contains(fb.Note, "new") {
		t.Errorf("note should flag this as new: %q", fb.Note)
	}
	if !strings.Contains(cmp.Verdict, "GPU->CPU fallback") && !strings.Contains(cmp.Verdict, "fallback") {
		t.Errorf("verdict should surface the new fallback pattern: %q", cmp.Verdict)
	}
}

func TestCompareTraces_PatternResolved(t *testing.T) {
	diag := New(nil, nil)
	c := NewComparer(nil, diag)
	var before, after []trace.Trace
	for i := 0; i < 10; i++ {
		before = append(before, mkTrace("b", 400, 100, "", trace.Context{ConcurrentTraces: 1}))
		after = append(after, mkTrace("a", 400, 100, "", trace.Context{ConcurrentTraces: 0}))
	}
	cmp := c.CompareTraces("", nil, nil, before, after)
	var cont *PatternDelta
	for i := range cmp.Patterns {
		if cmp.Patterns[i].Kind == KindContention {
			cont = &cmp.Patterns[i]
		}
	}
	if cont == nil || cont.BeforeCount != 10 || cont.AfterCount != 0 {
		t.Fatalf("contention pattern: %+v", cont)
	}
	if !strings.Contains(cont.Note, "resolved") {
		t.Errorf("note should say resolved: %q", cont.Note)
	}
}

func TestCompareTraces_NoDiagnoser_PatternsEmpty(t *testing.T) {
	c := NewComparer(nil, nil) // no diagnoser wired
	before := []trace.Trace{mkTrace("b", 400, 100, "", trace.Context{ConcurrentTraces: 5})}
	after := []trace.Trace{mkTrace("a", 400, 100, "", trace.Context{ConcurrentTraces: 5})}
	cmp := c.CompareTraces("", nil, nil, before, after)
	if len(cmp.Patterns) != 0 {
		t.Errorf("without a diagnoser, patterns should be empty: %+v", cmp.Patterns)
	}
}

func TestCompareTraces_EmptySetsReportNotEnoughData(t *testing.T) {
	c := NewComparer(nil, nil)
	cmp := c.CompareTraces("x", nil, nil, nil, []trace.Trace{mkTrace("a", 1, 1, "", trace.Context{})})
	if !strings.Contains(cmp.Verdict, "Not enough data") {
		t.Errorf("verdict: %q", cmp.Verdict)
	}
}

// ---- Compare() wrapper against a fake store ----

type fakeLister struct {
	byTag map[string][]trace.Trace // keyed by "k=v"
}

func (f *fakeLister) ListTraces(q trace.Query) ([]trace.Trace, error) {
	for k, v := range q.Tags {
		return f.byTag[k+"="+v], nil
	}
	return nil, nil
}

func TestCompare_UsesTagSelectorsToPartitionTraces(t *testing.T) {
	store := &fakeLister{byTag: map[string][]trace.Trace{
		"version=v1": {mkTrace("b1", 400, 100, "", trace.Context{}), mkTrace("b2", 420, 95, "", trace.Context{})},
		"version=v2": {mkTrace("a1", 250, 150, "", trace.Context{})},
	}}
	c := NewComparer(store, nil)
	res, err := c.Compare(map[string]string{"version": "v1"}, map[string]string{"version": "v2"}, "", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	cmp := res.(Comparison)
	if cmp.Before.Count != 2 || cmp.After.Count != 1 {
		t.Fatalf("counts: %+v", cmp)
	}
}
