package core

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockAdapter returns canned snapshots and lets tests toggle availability
// and inject latency.
type mockAdapter struct {
	name  string
	delay time.Duration
	polls atomic.Int64

	mu    sync.Mutex
	snaps []RuntimeSnapshot
	err   error
}

func (m *mockAdapter) Name() string { return m.name }
func (m *mockAdapter) Poll(ctx context.Context) ([]RuntimeSnapshot, error) {
	m.polls.Add(1)
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.err != nil {
		return nil, m.err
	}
	out := make([]RuntimeSnapshot, len(m.snaps))
	copy(out, m.snaps)
	return out, nil
}
func (m *mockAdapter) set(snaps []RuntimeSnapshot, err error) {
	m.mu.Lock()
	m.snaps, m.err = snaps, err
	m.mu.Unlock()
}

func snap(rt, model, dev string) RuntimeSnapshot {
	return RuntimeSnapshot{Runtime: rt, Model: model, BackendDevice: dev, MemTotalMB: 1000, MemUsedMB: 500}
}

func countKind(evs []Event, k EventKind) int {
	n := 0
	for _, e := range evs {
		if e.Kind == k {
			n++
		}
	}
	return n
}

// A slow adapter must not delay a fast one: with independent goroutines the
// fast adapter is polled many times while the slow one completes once.
func TestAdaptersDoNotBlockEachOther(t *testing.T) {
	slow := &mockAdapter{name: "slow", delay: 300 * time.Millisecond}
	slow.set([]RuntimeSnapshot{snap("slow", "m", "cpu")}, nil)
	fast := &mockAdapter{name: "fast"}
	fast.set([]RuntimeSnapshot{snap("fast", "m", "mps")}, nil)

	c := NewCollector()
	c.Logger = nil
	c.Register(slow, 50*time.Millisecond)
	c.Register(fast, 10*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	c.Run(ctx)

	fp, sp := fast.polls.Load(), slow.polls.Load()
	if fp < 15 {
		t.Fatalf("fast adapter polled only %d times; slow adapter (%d polls) blocked it", fp, sp)
	}
	if sp > 3 {
		t.Fatalf("slow adapter polled %d times, expected <= 3 given 300ms latency", sp)
	}
	keys := c.Rings()
	if len(keys) != 2 || keys[0] != "fast/m" || keys[1] != "slow/m" {
		t.Fatalf("unexpected ring keys %v", keys)
	}
}

// Snapshots from two adapters interleave into separate rings and Snapshot()
// returns the latest of each.
func TestInterleavingAndSnapshot(t *testing.T) {
	a := &mockAdapter{name: "a"}
	b := &mockAdapter{name: "b"}
	c := NewCollector()
	c.Logger = nil
	c.Register(a, time.Second)
	c.Register(b, time.Second)

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		sa := snap("a", "x", "mps")
		sa.TokensPerSec = float64(i)
		sb := snap("b", "y", "cpu")
		sb.TokensPerSec = float64(i * 10)
		a.set([]RuntimeSnapshot{sa}, nil)
		b.set([]RuntimeSnapshot{sb}, nil)
		c.PollNow(ctx)
	}
	got := c.Snapshot()
	if len(got) != 2 {
		t.Fatalf("want 2 snapshots, got %d", len(got))
	}
	if got[0].Key() != "a/x" || got[0].TokensPerSec != 4 {
		t.Errorf("a/x latest wrong: %+v", got[0])
	}
	if got[1].Key() != "b/y" || got[1].TokensPerSec != 40 {
		t.Errorf("b/y latest wrong: %+v", got[1])
	}
	if h := c.History("a", "x", time.Time{}, time.Now().Add(time.Hour)); len(h) != 5 {
		t.Errorf("history len = %d, want 5", len(h))
	}
}

func TestEventsRuntimeDownUpAndModelLifecycle(t *testing.T) {
	a := &mockAdapter{name: "ollama"}
	a.set(nil, fmt.Errorf("%w: connection refused", ErrRuntimeUnavailable))
	c := NewCollector()
	c.Logger = nil
	var got []Event
	var mu sync.Mutex
	c.OnEvent = func(e Event) { mu.Lock(); got = append(got, e); mu.Unlock() }
	c.Register(a, time.Second)
	ctx := context.Background()

	c.PollNow(ctx) // down
	c.PollNow(ctx) // still down: no duplicate event
	a.set([]RuntimeSnapshot{snap("ollama", "llama", "mps")}, nil)
	c.PollNow(ctx) // up + model loaded
	a.set([]RuntimeSnapshot{snap("ollama", "llama", "cpu")}, nil)
	c.PollNow(ctx) // backend changed
	a.set([]RuntimeSnapshot{}, nil)
	c.PollNow(ctx) // model unloaded
	a.set(nil, ErrRuntimeUnavailable)
	c.PollNow(ctx) // down again

	mu.Lock()
	defer mu.Unlock()
	want := map[EventKind]int{
		EventRuntimeDown: 2, EventRuntimeUp: 1, EventModelLoaded: 1,
		EventBackendChanged: 1, EventModelUnloaded: 1,
	}
	for k, n := range want {
		if got := countKind(got, k); got != n {
			t.Errorf("%s: got %d events, want %d (all: %+v)", k, got, n, got)
		}
	}
	st := c.Status()
	if len(st) != 1 || st[0].Up {
		t.Errorf("status should show runtime down: %+v", st)
	}
}

func TestNonAvailabilityErrorDoesNotUnloadModels(t *testing.T) {
	a := &mockAdapter{name: "vllm"}
	a.set([]RuntimeSnapshot{snap("vllm", "m", "cpu")}, nil)
	c := NewCollector()
	c.Logger = nil
	c.Register(a, time.Second)
	c.PollNow(context.Background())
	a.set(nil, errors.New("parse error"))
	c.PollNow(context.Background())
	if n := countKind(c.Events(), EventPollError); n != 1 {
		t.Errorf("want 1 poll_error event, got %d", n)
	}
	if n := countKind(c.Events(), EventModelUnloaded); n != 0 {
		t.Errorf("transient parse error must not unload models")
	}
}

func TestPollTimeoutBoundsHungAdapter(t *testing.T) {
	a := &mockAdapter{name: "hung", delay: 10 * time.Second}
	c := NewCollector()
	c.Logger = nil
	c.PollTimeout = 50 * time.Millisecond
	c.Register(a, time.Second)
	start := time.Now()
	c.PollNow(context.Background())
	if time.Since(start) > time.Second {
		t.Fatal("poll was not bounded by PollTimeout")
	}
}

func TestThroughputEnrichment(t *testing.T) {
	a := &mockAdapter{name: "ollama"}
	a.set([]RuntimeSnapshot{snap("ollama", "m", "mps")}, nil)
	c := NewCollector()
	c.Logger = nil
	c.Register(a, time.Second)
	c.ObserveThroughput("ollama", "m", 42.5, 1200)
	c.PollNow(context.Background())
	s := c.Snapshot()[0]
	if s.TokensPerSec != 42.5 || s.TPSSource != "traces" || s.ContextUsedTok != 1200 {
		t.Errorf("enrichment failed: %+v", s)
	}
}

// Concurrent readers during polling must be race-free (run with -race).
func TestConcurrentReadsDuringPolling(t *testing.T) {
	a := &mockAdapter{name: "a"}
	a.set([]RuntimeSnapshot{snap("a", "m", "mps")}, nil)
	c := NewCollector()
	c.Logger = nil
	c.Register(a, time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				_ = c.Snapshot()
				_ = c.Current()
				_ = c.Events()
				_ = c.Status()
			}
		}()
	}
	c.Run(ctx)
	wg.Wait()
}

func TestRing(t *testing.T) {
	r := NewRing(3)
	base := time.Now()
	for i := 0; i < 5; i++ {
		r.Push(RuntimeSnapshot{TokensPerSec: float64(i), Timestamp: base.Add(time.Duration(i) * time.Second)})
	}
	if r.Len() != 3 {
		t.Fatalf("len=%d", r.Len())
	}
	all := r.All()
	if all[0].TokensPerSec != 2 || all[2].TokensPerSec != 4 {
		t.Errorf("order wrong: %+v", all)
	}
	last, _ := r.Last()
	if last.TokensPerSec != 4 {
		t.Errorf("last wrong")
	}
	got := r.Range(base.Add(3*time.Second), base.Add(3*time.Second))
	if len(got) != 1 || got[0].TokensPerSec != 3 {
		t.Errorf("range wrong: %+v", got)
	}
}
