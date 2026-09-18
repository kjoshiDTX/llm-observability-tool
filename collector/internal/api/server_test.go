package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/storage"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

func newTestServer(t *testing.T) (*httptest.Server, *core.Collector) {
	t.Helper()
	st, err := storage.Open(t.TempDir() + "/api.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	c := core.NewCollector()
	c.Logger = nil
	s := &Server{Collector: c, Store: st, Ingester: trace.NewIngester(st, c)}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv, c
}

func post(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestTraceIngestAndStats(t *testing.T) {
	srv, _ := newTestServer(t)
	now := time.Now()
	for i := 1; i <= 20; i++ {
		body := map[string]any{
			"id": "t" + string(rune('a'+i)), "name": "summarize", "model": "m", "runtime": "ollama",
			"start_time": now.Add(-time.Duration(i) * time.Second), "end_time": now.Add(-time.Duration(i)*time.Second + time.Duration(i*10)*time.Millisecond),
			"completion_tokens": 10, "tags": map[string]string{"version": "v1"},
			"context": map[string]any{"mem_pressure": 0.99}, // must be ignored
		}
		resp := post(t, srv.URL+"/traces", body)
		if resp.StatusCode != 201 {
			t.Fatalf("status %d", resp.StatusCode)
		}
		var stored trace.Trace
		_ = json.NewDecoder(resp.Body).Decode(&stored)
		resp.Body.Close()
		if stored.Context.MemPressure != 0 {
			t.Fatal("client-supplied context was trusted")
		}
	}
	resp, err := http.Get(srv.URL + "/traces?name=summarize&window=1h")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Groups []trace.Stats `json:"groups"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Groups) != 1 || out.Groups[0].Count != 20 {
		t.Fatalf("stats: %+v", out)
	}
	g := out.Groups[0]
	if g.P50MS != 100 || g.P95MS != 190 || g.P99MS != 200 {
		t.Errorf("percentiles: %+v", g)
	}
	// Tag filter
	resp2, _ := http.Get(srv.URL + "/traces/list?tag=version=v2")
	var lst []trace.Trace
	_ = json.NewDecoder(resp2.Body).Decode(&lst)
	resp2.Body.Close()
	if len(lst) != 0 {
		t.Errorf("tag filter should exclude all: %d", len(lst))
	}
	// Get by id + 404
	if r, _ := http.Get(srv.URL + "/traces/tb"); r.StatusCode != 200 {
		t.Errorf("get by id: %d", r.StatusCode)
	}
	if r, _ := http.Get(srv.URL + "/traces/nope"); r.StatusCode != 404 {
		t.Errorf("missing id: %d", r.StatusCode)
	}
	if r, _ := http.Get(srv.URL + "/traces/nope/diagnosis"); r.StatusCode != 501 {
		t.Errorf("diagnosis without diagnoser: %d", r.StatusCode)
	}
}

func TestBadInput(t *testing.T) {
	srv, _ := newTestServer(t)
	resp, _ := http.Post(srv.URL+"/traces", "application/json", bytes.NewReader([]byte("{not json")))
	if resp.StatusCode != 400 {
		t.Errorf("bad json: %d", resp.StatusCode)
	}
	resp = post(t, srv.URL+"/traces", map[string]any{"name": "x"})
	if resp.StatusCode != 400 {
		t.Errorf("missing id: %d", resp.StatusCode)
	}
	resp, _ = http.Get(srv.URL + "/snapshots")
	if resp.StatusCode != 400 {
		t.Errorf("snapshots without key: %d", resp.StatusCode)
	}
}

func TestHealthAndSnapshot(t *testing.T) {
	srv, _ := newTestServer(t)
	resp, _ := http.Get(srv.URL + "/health")
	if resp.StatusCode != 200 {
		t.Fatal("health")
	}
	resp, _ = http.Get(srv.URL + "/snapshot")
	var snaps []core.RuntimeSnapshot
	_ = json.NewDecoder(resp.Body).Decode(&snaps)
	if snaps == nil {
		t.Error("snapshot should be [] not null")
	}
}
