package export

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

func scrape(t *testing.T, e *Exporter) string {
	t.Helper()
	rec := httptest.NewRecorder()
	e.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	return rec.Body.String()
}

func TestExporterSnapshotTraceAndUnload(t *testing.T) {
	e := New()
	e.ObserveSnapshot(core.RuntimeSnapshot{Runtime: "ollama", Model: "m", BackendDevice: "mps", TokensPerSec: 42, ModelMemMB: 1400, MemUsedMB: 10, MemTotalMB: 16, Timestamp: time.Now()})
	e.ObserveTrace(trace.Trace{Runtime: "ollama", Model: "m", Name: "summarize", DurationMS: 250, CompletionTokens: 20, TokensPerSec: 80})
	e.ObserveEvent(core.Event{Kind: core.EventModelLoaded, Runtime: "ollama", Model: "m"})
	body := scrape(t, e)
	for _, want := range []string{
		`llm_obs_tokens_per_sec{device="mps",model="m",runtime="ollama"} 42`,
		`llm_obs_model_loaded{device="mps",model="m",runtime="ollama"} 1`,
		`llm_obs_traces_total{model="m",name="summarize",runtime="ollama",status="ok"} 1`,
		`llm_obs_trace_duration_seconds_bucket{model="m",name="summarize",runtime="ollama",le="0.25"} 1`,
		`llm_obs_events_total{kind="model_loaded",runtime="ollama"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q", want)
		}
	}
	// Device change replaces the series rather than adding a second one.
	e.ObserveSnapshot(core.RuntimeSnapshot{Runtime: "ollama", Model: "m", BackendDevice: "cpu", TokensPerSec: 5, Timestamp: time.Now()})
	body = scrape(t, e)
	if strings.Contains(body, `device="mps"`) || !strings.Contains(body, `llm_obs_tokens_per_sec{device="cpu",model="m",runtime="ollama"} 5`) {
		t.Error("stale device series not replaced")
	}
	// Unload clears gauges.
	e.ObserveEvent(core.Event{Kind: core.EventModelUnloaded, Runtime: "ollama", Model: "m"})
	body = scrape(t, e)
	if strings.Contains(body, "llm_obs_model_loaded{") {
		t.Error("model_loaded gauge not cleared on unload")
	}
	// Counters survive.
	if !strings.Contains(body, `llm_obs_traces_total{model="m",name="summarize",runtime="ollama",status="ok"} 1`) {
		t.Error("counter lost")
	}
}
