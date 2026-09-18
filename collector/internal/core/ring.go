package core

import (
	"sync"
	"time"
)

// Ring is a fixed-capacity, thread-safe circular buffer of snapshots kept per
// (runtime, model). ~300 samples at a 2s poll is ten minutes of history,
// enough for rate/derivative calculations and trace correlation.
type Ring struct {
	mu   sync.RWMutex
	buf  []RuntimeSnapshot
	head int // next write position
	n    int // number of valid entries
}

// DefaultRingSize is the per-key buffer capacity.
const DefaultRingSize = 300

// NewRing returns a ring with the given capacity (min 1).
func NewRing(capacity int) *Ring {
	if capacity < 1 {
		capacity = 1
	}
	return &Ring{buf: make([]RuntimeSnapshot, capacity)}
}

// Push appends a snapshot, overwriting the oldest when full.
func (r *Ring) Push(s RuntimeSnapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf[r.head] = s
	r.head = (r.head + 1) % len(r.buf)
	if r.n < len(r.buf) {
		r.n++
	}
}

// Len returns the number of stored snapshots.
func (r *Ring) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.n
}

// Last returns the most recent snapshot.
func (r *Ring) Last() (RuntimeSnapshot, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.n == 0 {
		return RuntimeSnapshot{}, false
	}
	idx := (r.head - 1 + len(r.buf)) % len(r.buf)
	return r.buf[idx], true
}

// All returns a copy of the snapshots in chronological order.
func (r *Ring) All() []RuntimeSnapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]RuntimeSnapshot, 0, r.n)
	start := (r.head - r.n + len(r.buf)) % len(r.buf)
	for i := 0; i < r.n; i++ {
		out = append(out, r.buf[(start+i)%len(r.buf)])
	}
	return out
}

// Range returns snapshots with from <= Timestamp <= to, in order.
func (r *Ring) Range(from, to time.Time) []RuntimeSnapshot {
	all := r.All()
	out := all[:0]
	for _, s := range all {
		if !s.Timestamp.Before(from) && !s.Timestamp.After(to) {
			out = append(out, s)
		}
	}
	return out
}

// Rate returns the per-second change of a numeric field between the oldest
// and newest snapshot inside the window, or 0 if fewer than two samples.
func (r *Ring) Rate(window time.Duration, field func(RuntimeSnapshot) float64) float64 {
	now := time.Now()
	pts := r.Range(now.Add(-window), now)
	if len(pts) < 2 {
		return 0
	}
	first, last := pts[0], pts[len(pts)-1]
	dt := last.Timestamp.Sub(first.Timestamp).Seconds()
	if dt <= 0 {
		return 0
	}
	return (field(last) - field(first)) / dt
}
