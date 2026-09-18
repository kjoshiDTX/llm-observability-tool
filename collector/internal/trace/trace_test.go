package trace

import (
	"context"
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
)

type memStore struct {
	traces []Trace
	snaps  []core.RuntimeSnapshot
}

func (m *memStore) InsertTrace(t Trace) error { m.traces = append(m.traces, t); return nil }
func (m *memStore) GetTrace(id string) (Trace, bool, error) {
	for _, t := range m.traces {
		if t.ID == id {
			return t, true, nil
		}
	}
	return Trace{}, false, nil
}
func (m *memStore) ListTraces(q Query) ([]Trace, error) { return m.traces, nil }
func (m *memStore) OverlappingTraces(model string, start, end time.Time, ex string) ([]Trace, error) {
	var out []Trace
	for _, t := range m.traces {
		if t.ID != ex && t.Model == model && !t.StartTime.After(end) && !t.EndTime.Before(start) {
			out = append(out, t)
		}
	}
	return out, nil
}
func (m *memStore) SnapshotsInRange(rt, model string, from, to time.Time) ([]core.RuntimeSnapshot, error) {
	return m.snaps, nil
}

type fakeSrc struct {
	hist []core.RuntimeSnapshot
	tps  float64
}

func (f *fakeSrc) History(rt, model string, from, to time.Time) []core.RuntimeSnapshot { return f.hist }
func (f *fakeSrc) RuntimeFor(model string) string                                      { return "ollama" }
func (f *fakeSrc) ObserveThroughput(rt, model string, tps float64, ctx int)            { f.tps = tps }

func TestIngestEnrichesAndComputesTPS(t *testing.T) {
	now := time.Now()
	src := &fakeSrc{hist: []core.RuntimeSnapshot{
		{Runtime: "ollama", Model: "m", BackendDevice: "mps", MemUsedMB: 8000, MemTotalMB: 16000, ContextMaxTok: 4096, KVCacheUsedPct: 10, Timestamp: now},
		{Runtime: "ollama", Model: "m", BackendDevice: "cpu", MemUsedMB: 15000, MemTotalMB: 16000, ContextMaxTok: 4096, KVCacheUsedPct: 95, Timestamp: now.Add(time.Second)},
	}}
	st := &memStore{}
	in := NewIngester(st, src)
	got, err := in.Ingest(context.Background(), Trace{ID: "t1", Name: "n", Model: "m", StartTime: now, EndTime: now.Add(2 * time.Second),
		PromptTokens: 3000, CompletionTokens: 100, TTFTMS: 500})
	if err != nil {
		t.Fatal(err)
	}
	if got.Runtime != "ollama" {
		t.Errorf("runtime not resolved: %q", got.Runtime)
	}
	if got.DurationMS != 2000 {
		t.Errorf("duration = %v", got.DurationMS)
	}
	// 100 tokens over (2000-500)ms = 66.67 tok/s
	if got.TokensPerSec < 66 || got.TokensPerSec > 67 {
		t.Errorf("tps = %v", got.TokensPerSec)
	}
	c := got.Context
	if c.SnapshotCount != 2 || c.KVCacheMaxPct != 95 || c.BackendDevice != "cpu" || len(c.BackendDevices) != 2 {
		t.Errorf("context wrong: %+v", c)
	}
	if c.ContextUsedTok != 3100 || c.ContextMaxTok != 4096 || c.ContextPressure < 0.75 {
		t.Errorf("context pressure wrong: %+v", c)
	}
	if c.MemPressure < 0.93 {
		t.Errorf("mem pressure wrong: %v", c.MemPressure)
	}
	if src.tps == 0 {
		t.Error("throughput not fed back to collector")
	}
	if len(st.traces) != 1 {
		t.Error("trace not stored")
	}
}

func TestIngestValidation(t *testing.T) {
	in := NewIngester(&memStore{}, nil)
	if _, err := in.Ingest(context.Background(), Trace{Name: "n", StartTime: time.Now()}); err == nil {
		t.Error("missing id accepted")
	}
	if _, err := in.Ingest(context.Background(), Trace{ID: "x", StartTime: time.Now()}); err == nil {
		t.Error("missing name accepted")
	}
	if _, err := in.Ingest(context.Background(), Trace{ID: "x", Name: "n"}); err == nil {
		t.Error("missing start accepted")
	}
	now := time.Now()
	if _, err := in.Ingest(context.Background(), Trace{ID: "x", Name: "n", StartTime: now, EndTime: now.Add(-time.Second)}); err == nil {
		t.Error("end before start accepted")
	}
}

func TestConcurrencyCountsInflightAndStored(t *testing.T) {
	now := time.Now()
	st := &memStore{}
	in := NewIngester(st, nil)
	// A completed overlapping trace on the same model.
	_, _ = in.Ingest(context.Background(), Trace{ID: "done", Name: "n", Model: "m", StartTime: now, EndTime: now.Add(time.Second)})
	// An in-flight one on the same model, and one on a different model.
	_ = in.BeginTrace(Begin{ID: "inflight", Model: "m", StartTime: now.Add(200 * time.Millisecond)})
	_ = in.BeginTrace(Begin{ID: "other", Model: "z", StartTime: now})
	got, _ := in.Ingest(context.Background(), Trace{ID: "me", Name: "n", Model: "m", StartTime: now.Add(100 * time.Millisecond), EndTime: now.Add(900 * time.Millisecond)})
	if got.Context.ConcurrentTraces != 2 {
		t.Errorf("concurrent = %d, want 2", got.Context.ConcurrentTraces)
	}
	// Non-overlapping later trace: none.
	got, _ = in.Ingest(context.Background(), Trace{ID: "later", Name: "n", Model: "m", StartTime: now.Add(5 * time.Second), EndTime: now.Add(6 * time.Second)})
	if got.Context.ConcurrentTraces != 1 { // "inflight" has no end yet and started before, so it still counts
		t.Errorf("later concurrent = %d, want 1", got.Context.ConcurrentTraces)
	}
}

func TestPercentilesAndAggregate(t *testing.T) {
	vals := []float64{5, 1, 4, 2, 3}
	if Percentile(vals, 50) != 3 || Percentile(vals, 99) != 5 || Percentile(vals, 0) != 1 {
		t.Errorf("percentiles wrong: %v %v %v", Percentile(vals, 50), Percentile(vals, 99), Percentile(vals, 0))
	}
	if Percentile(nil, 50) != 0 {
		t.Error("empty percentile")
	}
	var ts []Trace
	for i := 1; i <= 100; i++ {
		ts = append(ts, Trace{Name: "a", Model: "m", DurationMS: float64(i), TokensPerSec: 10})
	}
	ts = append(ts, Trace{Name: "b", Model: "m", DurationMS: 7, Error: "boom"})
	st := Aggregate(ts)
	if len(st) != 2 || st[0].Name != "a" || st[1].Name != "b" {
		t.Fatalf("groups: %+v", st)
	}
	a := st[0]
	if a.Count != 100 || a.P50MS != 50 || a.P95MS != 95 || a.P99MS != 99 || a.MeanMS != 50.5 || a.TPSMean != 10 {
		t.Errorf("stats wrong: %+v", a)
	}
	if st[1].Errors != 1 {
		t.Errorf("error count wrong")
	}
}

func TestParseTagSelector(t *testing.T) {
	if m := ParseTagSelector("version=v1", "x"); m["version"] != "v1" {
		t.Error("k=v")
	}
	if m := ParseTagSelector("v1", "version"); m["version"] != "v1" {
		t.Error("bare value")
	}
	if ParseTagSelector("", "x") != nil {
		t.Error("empty")
	}
}
