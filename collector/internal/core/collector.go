package core

import (
	"context"
	"errors"
	"log"
	"sort"
	"sync"
	"time"
)

// Registration pairs an adapter with its poll interval.
type Registration struct {
	Adapter  Adapter
	Interval time.Duration
}

// Collector polls every registered adapter on its own goroutine, fans the
// resulting snapshots into per-(runtime, model) ring buffers, detects state
// transitions (runtime up/down, model load/unload, backend change), and
// exposes a thread-safe read view.
type Collector struct {
	regs     []Registration
	ringSize int

	mu       sync.RWMutex
	rings    map[string]*Ring    // key -> ring
	runtimes map[string]*rtState // runtime name -> state
	tps      map[string]tpsObs   // key -> recent throughput seen via traces
	events   []Event             // bounded recent events
	eventCap int

	// Hooks are invoked synchronously on the polling goroutine; keep them fast
	// (storage writes should be batched or buffered by the hook itself).
	OnSnapshot func(RuntimeSnapshot)
	OnEvent    func(Event)

	// PollTimeout bounds a single adapter Poll so one hung runtime cannot
	// stall its own loop indefinitely.
	PollTimeout time.Duration
	Logger      *log.Logger
}

type rtState struct {
	up       bool
	known    bool              // have we completed at least one poll?
	models   map[string]string // model -> backend device
	lastPoll time.Time
	lastErr  string
}

type tpsObs struct {
	tps     float64
	ctxUsed int
	at      time.Time
}

// NewCollector creates a collector with default settings.
func NewCollector() *Collector {
	return &Collector{
		ringSize:    DefaultRingSize,
		rings:       map[string]*Ring{},
		runtimes:    map[string]*rtState{},
		tps:         map[string]tpsObs{},
		eventCap:    500,
		PollTimeout: 5 * time.Second,
		Logger:      log.Default(),
	}
}

// Register adds an adapter polled at the given interval.
func (c *Collector) Register(a Adapter, interval time.Duration) {
	if interval <= 0 {
		interval = 2 * time.Second
	}
	c.regs = append(c.regs, Registration{Adapter: a, Interval: interval})
	c.mu.Lock()
	c.runtimes[a.Name()] = &rtState{models: map[string]string{}}
	c.mu.Unlock()
}

// SetRingSize overrides the per-key buffer capacity (before Run).
func (c *Collector) SetRingSize(n int) { c.ringSize = n }

// Run blocks, polling all adapters until ctx is cancelled, then waits for
// every polling goroutine to exit.
func (c *Collector) Run(ctx context.Context) {
	var wg sync.WaitGroup
	for _, reg := range c.regs {
		wg.Add(1)
		go func(reg Registration) {
			defer wg.Done()
			c.loop(ctx, reg)
		}(reg)
	}
	wg.Wait()
}

func (c *Collector) loop(ctx context.Context, reg Registration) {
	t := time.NewTicker(reg.Interval)
	defer t.Stop()
	for {
		c.pollOnce(ctx, reg.Adapter)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// pollOnce runs a single poll cycle for one adapter. It is exported for
// tests via PollNow.
func (c *Collector) pollOnce(ctx context.Context, a Adapter) {
	pctx, cancel := context.WithTimeout(ctx, c.PollTimeout)
	snaps, err := a.Poll(pctx)
	cancel()
	now := time.Now()

	c.mu.Lock()
	st := c.runtimes[a.Name()]
	if st == nil {
		st = &rtState{models: map[string]string{}}
		c.runtimes[a.Name()] = st
	}
	st.lastPoll = now
	var evs []Event

	if err != nil {
		if errors.Is(err, ErrRuntimeUnavailable) {
			if st.up || !st.known {
				evs = append(evs, Event{Kind: EventRuntimeDown, Runtime: a.Name(), Detail: err.Error(), Timestamp: now})
			}
			st.up = false
			// Everything that was loaded is now gone.
			for m := range st.models {
				evs = append(evs, Event{Kind: EventModelUnloaded, Runtime: a.Name(), Model: m, Detail: "runtime went down", Timestamp: now})
			}
			st.models = map[string]string{}
		} else {
			if st.lastErr != err.Error() {
				evs = append(evs, Event{Kind: EventPollError, Runtime: a.Name(), Detail: err.Error(), Timestamp: now})
			}
		}
		st.lastErr = err.Error()
		st.known = true
		c.mu.Unlock()
		c.emit(evs)
		return
	}
	st.lastErr = ""
	if !st.up {
		evs = append(evs, Event{Kind: EventRuntimeUp, Runtime: a.Name(), Timestamp: now})
	}
	st.up = true
	st.known = true

	seen := map[string]bool{}
	for i := range snaps {
		s := &snaps[i]
		if s.Runtime == "" {
			s.Runtime = a.Name()
		}
		if s.Timestamp.IsZero() {
			s.Timestamp = now
		}
		seen[s.Model] = true
		prevDev, existed := st.models[s.Model]
		if !existed {
			evs = append(evs, Event{Kind: EventModelLoaded, Runtime: s.Runtime, Model: s.Model, Detail: "backend=" + s.BackendDevice, Timestamp: now})
		} else if prevDev != s.BackendDevice && s.BackendDevice != "" {
			evs = append(evs, Event{Kind: EventBackendChanged, Runtime: s.Runtime, Model: s.Model, Detail: prevDev + "->" + s.BackendDevice, Timestamp: now})
		}
		st.models[s.Model] = s.BackendDevice

		// Enrich from recent trace-derived throughput if the runtime itself
		// does not report it.
		if s.TokensPerSec == 0 {
			if o, ok := c.tps[s.Key()]; ok && now.Sub(o.at) < 30*time.Second {
				s.TokensPerSec = o.tps
				s.TPSSource = "traces"
				if s.ContextUsedTok == 0 {
					s.ContextUsedTok = o.ctxUsed
				}
			}
		} else if s.TPSSource == "" {
			s.TPSSource = "runtime"
		}
		r := c.rings[s.Key()]
		if r == nil {
			r = NewRing(c.ringSize)
			c.rings[s.Key()] = r
		}
		r.Push(*s)
	}
	for m := range st.models {
		if !seen[m] {
			evs = append(evs, Event{Kind: EventModelUnloaded, Runtime: a.Name(), Model: m, Timestamp: now})
			delete(st.models, m)
		}
	}
	c.mu.Unlock()

	if c.OnSnapshot != nil {
		for _, s := range snaps {
			c.OnSnapshot(s)
		}
	}
	c.emit(evs)
}

func (c *Collector) emit(evs []Event) {
	if len(evs) == 0 {
		return
	}
	c.mu.Lock()
	c.events = append(c.events, evs...)
	if len(c.events) > c.eventCap {
		c.events = c.events[len(c.events)-c.eventCap:]
	}
	c.mu.Unlock()
	for _, e := range evs {
		if c.Logger != nil {
			c.Logger.Printf("event %s runtime=%s model=%s %s", e.Kind, e.Runtime, e.Model, e.Detail)
		}
		if c.OnEvent != nil {
			c.OnEvent(e)
		}
	}
}

// PollNow forces a synchronous poll of every adapter (used by tests and by
// the API when a caller wants a fresh reading).
func (c *Collector) PollNow(ctx context.Context) {
	for _, reg := range c.regs {
		c.pollOnce(ctx, reg.Adapter)
	}
}

// ObserveThroughput records a trace-derived throughput reading so the next
// snapshot for that (runtime, model) can carry tokens/sec even when the
// runtime itself does not expose it (Ollama).
func (c *Collector) ObserveThroughput(runtime, model string, tps float64, ctxUsed int) {
	if tps <= 0 {
		return
	}
	c.mu.Lock()
	c.tps[runtime+"/"+model] = tpsObs{tps: tps, ctxUsed: ctxUsed, at: time.Now()}
	c.mu.Unlock()
}

// Snapshot returns the latest snapshot for every (runtime, model), sorted by
// key. Safe for concurrent use.
func (c *Collector) Snapshot() []RuntimeSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	keys := make([]string, 0, len(c.rings))
	for k := range c.rings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]RuntimeSnapshot, 0, len(keys))
	for _, k := range keys {
		if s, ok := c.rings[k].Last(); ok {
			out = append(out, s)
		}
	}
	return out
}

// Current returns only snapshots for models the runtime currently reports as
// loaded (rings persist history for unloaded models).
func (c *Collector) Current() []RuntimeSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []RuntimeSnapshot
	for rt, st := range c.runtimes {
		for m := range st.models {
			if r, ok := c.rings[rt+"/"+m]; ok {
				if s, ok := r.Last(); ok {
					out = append(out, s)
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key() < out[j].Key() })
	return out
}

// History returns snapshots for one key inside [from, to].
func (c *Collector) History(runtime, model string, from, to time.Time) []RuntimeSnapshot {
	c.mu.RLock()
	r := c.rings[runtime+"/"+model]
	c.mu.RUnlock()
	if r == nil {
		return nil
	}
	return r.Range(from, to)
}

// Rings returns the keys of every ring (for tests/debug).
func (c *Collector) Rings() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	keys := make([]string, 0, len(c.rings))
	for k := range c.rings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Events returns recent events, oldest first.
func (c *Collector) Events() []Event {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]Event(nil), c.events...)
}

// RuntimeStatus is the health view of one runtime.
type RuntimeStatus struct {
	Runtime  string    `json:"runtime"`
	Up       bool      `json:"up"`
	Models   []string  `json:"models"`
	LastPoll time.Time `json:"last_poll"`
	LastErr  string    `json:"last_error,omitempty"`
}

// Status returns the health of every registered runtime.
func (c *Collector) Status() []RuntimeStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]RuntimeStatus, 0, len(c.runtimes))
	for name, st := range c.runtimes {
		models := make([]string, 0, len(st.models))
		for m := range st.models {
			models = append(models, m)
		}
		sort.Strings(models)
		out = append(out, RuntimeStatus{Runtime: name, Up: st.up, Models: models, LastPoll: st.lastPoll, LastErr: st.lastErr})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Runtime < out[j].Runtime })
	return out
}

// RuntimeFor returns the runtime that currently has the model loaded, or "".
func (c *Collector) RuntimeFor(model string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	names := make([]string, 0, len(c.runtimes))
	for n := range c.runtimes {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		if _, ok := c.runtimes[n].models[model]; ok {
			return n
		}
	}
	return ""
}
