// Package api exposes the collector's local HTTP API used by the Python
// client and the llm-obs CLI.
package api

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/storage"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// Server bundles the collector's components behind an http.Handler.
type Server struct {
	Collector *core.Collector
	Store     *storage.Store
	Ingester  *trace.Ingester
	Diagnoser Diagnoser
	Comparer  Comparer
	Checker   Checker
	Metrics   http.Handler // optional Prometheus exporter
	Logger    *log.Logger
	Version   string
}

// Diagnoser is implemented by core.Diagnoser (phase 5).
type Diagnoser interface {
	Diagnose(id string) (any, error)
}

// Comparer is implemented by core.Comparer (phase 6).
type Comparer interface {
	Compare(before, after map[string]string, name string, since time.Time) (any, error)
}

// Checker is implemented by core.Preflight (phase 7).
type Checker interface {
	Check(model string, contextTok int) (any, error)
}

// ErrNotFound is returned by Diagnoser/Comparer for unknown ids.
var ErrNotFound = errors.New("not found")

// Handler builds the router.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /snapshot", s.snapshot)
	mux.HandleFunc("GET /snapshots", s.snapshots)
	mux.HandleFunc("GET /events", s.events)
	mux.HandleFunc("POST /traces/begin", s.traceBegin)
	mux.HandleFunc("POST /traces", s.traceIngest)
	mux.HandleFunc("GET /traces", s.traceStats)
	mux.HandleFunc("GET /traces/list", s.traceList)
	mux.HandleFunc("GET /traces/{id}", s.traceGet)
	mux.HandleFunc("GET /traces/{id}/diagnosis", s.traceDiagnosis)
	mux.HandleFunc("GET /compare", s.compare)
	mux.HandleFunc("GET /check", s.check)
	if s.Metrics != nil {
		mux.Handle("GET /metrics", s.Metrics)
	}
	return logMiddleware(s.Logger, mux)
}

func logMiddleware(l *log.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				if l != nil {
					l.Printf("panic in %s %s: %v", r.Method, r.URL.Path, rec)
				}
				writeErr(w, http.StatusInternalServerError, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	resp := map[string]any{"status": "ok", "version": s.Version, "time": time.Now()}
	if s.Collector != nil {
		resp["runtimes"] = s.Collector.Status()
	}
	if s.Store != nil {
		sn, tr, ev := s.Store.Counts()
		resp["db"] = map[string]int64{"snapshots": sn, "traces": tr, "events": ev}
	}
	if s.Ingester != nil {
		resp["inflight_traces"] = s.Ingester.InFlight()
	}
	writeJSON(w, 200, resp)
}

func (s *Server) snapshot(w http.ResponseWriter, r *http.Request) {
	if s.Collector == nil {
		writeJSON(w, 200, []any{})
		return
	}
	if r.URL.Query().Get("fresh") == "1" {
		s.Collector.PollNow(r.Context())
	}
	if r.URL.Query().Get("all") == "1" {
		writeJSON(w, 200, s.Collector.Snapshot())
		return
	}
	snaps := s.Collector.Current()
	if snaps == nil {
		snaps = []core.RuntimeSnapshot{}
	}
	writeJSON(w, 200, snaps)
}

func (s *Server) snapshots(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	rt, model := q.Get("runtime"), q.Get("model")
	if rt == "" || model == "" {
		writeErr(w, 400, "runtime and model are required")
		return
	}
	since := parseDuration(q.Get("since"), 10*time.Minute)
	now := time.Now()
	var out []core.RuntimeSnapshot
	if s.Collector != nil {
		out = s.Collector.History(rt, model, now.Add(-since), now)
	}
	if len(out) == 0 && s.Store != nil {
		out, _ = s.Store.SnapshotsInRange(rt, model, now.Add(-since), now)
	}
	if out == nil {
		out = []core.RuntimeSnapshot{}
	}
	writeJSON(w, 200, out)
}

func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	since := parseDuration(r.URL.Query().Get("since"), time.Hour)
	var out []core.Event
	if s.Store != nil {
		out, _ = s.Store.EventsInRange(time.Now().Add(-since), time.Now(), r.URL.Query().Get("runtime"))
	} else if s.Collector != nil {
		out = s.Collector.Events()
	}
	if out == nil {
		out = []core.Event{}
	}
	writeJSON(w, 200, out)
}

func (s *Server) traceBegin(w http.ResponseWriter, r *http.Request) {
	var b trace.Begin
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&b); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if err := s.Ingester.BeginTrace(b); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 202, map[string]string{"id": b.ID})
}

func (s *Server) traceIngest(w http.ResponseWriter, r *http.Request) {
	var t trace.Trace
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&t); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	t.Context = trace.Context{} // never trust client-supplied context
	stored, err := s.Ingester.Ingest(r.Context(), t)
	if err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 201, stored)
}

func (s *Server) traceStats(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	window := parseDuration(q.Get("window"), time.Hour)
	tq := trace.Query{Name: q.Get("name"), Model: q.Get("model"), Since: time.Now().Add(-window), Tags: parseTags(q["tag"])}
	traces, err := s.Store.ListTraces(tq)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	stats := trace.Aggregate(traces)
	if stats == nil {
		stats = []trace.Stats{}
	}
	writeJSON(w, 200, map[string]any{"window": window.String(), "groups": stats})
}

func (s *Server) traceList(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit <= 0 {
		limit = 100
	}
	tq := trace.Query{Name: q.Get("name"), Model: q.Get("model"), Limit: limit, Tags: parseTags(q["tag"])}
	if w := q.Get("window"); w != "" {
		tq.Since = time.Now().Add(-parseDuration(w, time.Hour))
	}
	traces, err := s.Store.ListTraces(tq)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if traces == nil {
		traces = []trace.Trace{}
	}
	writeJSON(w, 200, traces)
}

func (s *Server) traceGet(w http.ResponseWriter, r *http.Request) {
	t, ok, err := s.Store.GetTrace(r.PathValue("id"))
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if !ok {
		writeErr(w, 404, "trace not found")
		return
	}
	writeJSON(w, 200, t)
}

func (s *Server) traceDiagnosis(w http.ResponseWriter, r *http.Request) {
	if s.Diagnoser == nil {
		writeErr(w, 501, "diagnoser not configured")
		return
	}
	d, err := s.Diagnoser.Diagnose(r.PathValue("id"))
	if errors.Is(err, ErrNotFound) || (err != nil && err.Error() == "trace not found") {
		writeErr(w, 404, "trace not found")
		return
	}
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, d)
}

func (s *Server) compare(w http.ResponseWriter, r *http.Request) {
	if s.Comparer == nil {
		writeErr(w, 501, "comparer not configured")
		return
	}
	q := r.URL.Query()
	key := q.Get("tag_key")
	if key == "" {
		key = "version"
	}
	before, after := trace.ParseTagSelector(q.Get("before"), key), trace.ParseTagSelector(q.Get("after"), key)
	if before == nil || after == nil {
		writeErr(w, 400, "before and after selectors are required (key=value or value)")
		return
	}
	var since time.Time
	if wnd := q.Get("window"); wnd != "" {
		since = time.Now().Add(-parseDuration(wnd, 24*time.Hour))
	}
	res, err := s.Comparer.Compare(before, after, q.Get("name"), since)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, res)
}

func (s *Server) check(w http.ResponseWriter, r *http.Request) {
	if s.Checker == nil {
		writeErr(w, 501, "checker not configured")
		return
	}
	q := r.URL.Query()
	model := q.Get("model")
	if model == "" {
		writeErr(w, 400, "model is required")
		return
	}
	ctxTok, _ := strconv.Atoi(q.Get("context"))
	res, err := s.Checker.Check(model, ctxTok)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, res)
}

func parseDuration(s string, def time.Duration) time.Duration {
	if s == "" {
		return def
	}
	if d, err := time.ParseDuration(s); err == nil && d > 0 {
		return d
	}
	if n, err := strconv.Atoi(s); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	return def
}

func parseTags(vals []string) map[string]string {
	if len(vals) == 0 {
		return nil
	}
	out := map[string]string{}
	for _, v := range vals {
		if k, val, ok := strings.Cut(v, "="); ok {
			out[k] = val
		}
	}
	return out
}
