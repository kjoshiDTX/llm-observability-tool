package storage

import (
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

func openTest(t *testing.T) *Store {
	t.Helper()
	s, err := Open(t.TempDir() + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestSnapshotsRoundTrip(t *testing.T) {
	s := openTest(t)
	base := time.Now().Add(-10 * time.Second).Truncate(time.Millisecond)
	for i := 0; i < 5; i++ {
		if err := s.InsertSnapshot(core.RuntimeSnapshot{Runtime: "ollama", Model: "m", TokensPerSec: float64(i), BackendDevice: "mps",
			UnifiedMemory: true, MemTotalMB: 16384, Timestamp: base.Add(time.Duration(i) * time.Second)}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := s.SnapshotsInRange("ollama", "m", base.Add(time.Second), base.Add(3*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].TokensPerSec != 1 || got[2].TokensPerSec != 3 || !got[0].UnifiedMemory {
		t.Fatalf("range wrong: %+v", got)
	}
	if err := s.Prune(0); err != nil {
		t.Fatal(err)
	}
	if n, _, _ := s.Counts(); n != 0 {
		t.Errorf("prune left %d snapshots", n)
	}
}

func TestTracesQueryAndTags(t *testing.T) {
	s := openTest(t)
	base := time.Now()
	mk := func(id, name, ver string, start time.Time, dur float64) trace.Trace {
		return trace.Trace{ID: id, Name: name, Model: "m", Runtime: "ollama", Tags: map[string]string{"version": ver},
			StartTime: start, EndTime: start.Add(time.Duration(dur) * time.Millisecond), DurationMS: dur, CompletionTokens: 10,
			Context: trace.Context{BackendDevice: "mps"}}
	}
	for _, tr := range []trace.Trace{
		mk("a", "sum", "v1", base, 100), mk("b", "sum", "v1", base.Add(time.Second), 200),
		mk("c", "sum", "v2", base.Add(2*time.Second), 300), mk("d", "chat", "v2", base.Add(3*time.Second), 400),
	} {
		if err := s.InsertTrace(tr); err != nil {
			t.Fatal(err)
		}
	}
	got, _ := s.ListTraces(trace.Query{Name: "sum"})
	if len(got) != 3 || got[0].ID != "c" {
		t.Errorf("name filter/order wrong: %+v", got)
	}
	got, _ = s.ListTraces(trace.Query{Tags: map[string]string{"version": "v2"}})
	if len(got) != 2 {
		t.Errorf("tag filter wrong: %d", len(got))
	}
	got, _ = s.ListTraces(trace.Query{Since: base.Add(1500 * time.Millisecond), Limit: 1})
	if len(got) != 1 || got[0].ID != "d" {
		t.Errorf("since/limit wrong: %+v", got)
	}
	one, ok, err := s.GetTrace("a")
	if err != nil || !ok || one.Context.BackendDevice != "mps" || one.Tags["version"] != "v1" {
		t.Errorf("get wrong: %+v %v %v", one, ok, err)
	}
	if _, ok, _ := s.GetTrace("nope"); ok {
		t.Error("missing trace reported found")
	}
	// Overlap: window [base+0.15s, base+1.1s] intersects a (0..0.1s)? no. b (1s..1.2s)? yes.
	ov, _ := s.OverlappingTraces("m", base.Add(150*time.Millisecond), base.Add(1100*time.Millisecond), "zzz")
	if len(ov) != 1 || ov[0].ID != "b" {
		t.Errorf("overlap wrong: %+v", ov)
	}
	bl, _ := s.Baseline("sum", "m", base.Add(2*time.Second), 10)
	if len(bl) != 2 {
		t.Errorf("baseline wrong: %d", len(bl))
	}
}

func TestEvents(t *testing.T) {
	s := openTest(t)
	now := time.Now()
	_ = s.InsertEvent(core.Event{Kind: core.EventRuntimeDown, Runtime: "ollama", Timestamp: now})
	_ = s.InsertEvent(core.Event{Kind: core.EventModelLoaded, Runtime: "vllm", Model: "m", Timestamp: now})
	ev, err := s.EventsInRange(now.Add(-time.Minute), now.Add(time.Minute), "vllm")
	if err != nil || len(ev) != 1 || ev[0].Kind != core.EventModelLoaded {
		t.Errorf("events: %+v %v", ev, err)
	}
}

func TestSnapshotHookBatches(t *testing.T) {
	s := openTest(t)
	ctx, cancel := contextWithCancel()
	s.StartWriter(ctx)
	hook := s.SnapshotHook()
	for i := 0; i < 600; i++ {
		hook(core.RuntimeSnapshot{Runtime: "r", Model: "m", Timestamp: time.Now()})
	}
	cancel()
	s.Close()
	s2, err := Open(s.path())
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	if n, _, _ := s2.Counts(); n != 600 {
		t.Errorf("expected 600 snapshots persisted, got %d", n)
	}
}
