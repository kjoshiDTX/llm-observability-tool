// Package export exposes collector data as Prometheus metrics with
// low-cardinality labels (runtime, model, device, trace name). Per-request
// ids are never used as labels.
package export

import (
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// Exporter holds the registry and metric vectors.
type Exporter struct {
	reg *prometheus.Registry

	tokensPerSec    *prometheus.GaugeVec
	memUsed         *prometheus.GaugeVec
	memTotal        *prometheus.GaugeVec
	modelMem        *prometheus.GaugeVec
	modelGPUMem     *prometheus.GaugeVec
	kvCache         *prometheus.GaugeVec
	ctxMax          *prometheus.GaugeVec
	reqRunning      *prometheus.GaugeVec
	reqWaiting      *prometheus.GaugeVec
	modelLoaded     *prometheus.GaugeVec
	events          *prometheus.CounterVec
	traces          *prometheus.CounterVec
	traceDuration   *prometheus.HistogramVec
	traceTokens     *prometheus.CounterVec
	traceTPS        *prometheus.GaugeVec
	traceConcurrent *prometheus.GaugeVec

	mu     sync.Mutex
	labels map[string]prometheus.Labels // key -> last label set, for cleanup on unload
}

// New builds an exporter with its own registry.
func New() *Exporter {
	reg := prometheus.NewRegistry()
	reg.MustRegister(prometheus.NewGoCollector(), prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	rm := []string{"runtime", "model"}
	rmd := []string{"runtime", "model", "device"}
	e := &Exporter{
		reg:          reg,
		labels:       map[string]prometheus.Labels{},
		tokensPerSec: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_tokens_per_sec", Help: "Generation throughput (runtime-reported or trace-derived)."}, rmd),
		memUsed:      prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_host_mem_used_mb", Help: "Host memory used (MB)."}, []string{"runtime"}),
		memTotal:     prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_host_mem_total_mb", Help: "Host memory total (MB)."}, []string{"runtime"}),
		modelMem:     prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_model_mem_mb", Help: "Resident model memory (MB)."}, rmd),
		modelGPUMem:  prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_model_gpu_mem_mb", Help: "Model memory on the accelerator (MB)."}, rmd),
		kvCache:      prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_kv_cache_used_pct", Help: "KV-cache utilisation percent (vLLM)."}, rm),
		ctxMax:       prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_context_max_tokens", Help: "Loaded context window size."}, rm),
		reqRunning:   prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_requests_running", Help: "Requests running per runtime report."}, rm),
		reqWaiting:   prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_requests_waiting", Help: "Requests queued per runtime report."}, rm),
		modelLoaded:  prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_model_loaded", Help: "1 while the model is resident."}, rmd),
		events:       prometheus.NewCounterVec(prometheus.CounterOpts{Name: "llm_obs_events_total", Help: "Collector state-change events."}, []string{"runtime", "kind"}),
		traces:       prometheus.NewCounterVec(prometheus.CounterOpts{Name: "llm_obs_traces_total", Help: "Traced LLM calls."}, []string{"runtime", "model", "name", "status"}),
		traceDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "llm_obs_trace_duration_seconds", Help: "Traced call latency.",
			Buckets: []float64{.05, .1, .25, .5, 1, 2, 5, 10, 20, 40, 80}}, []string{"runtime", "model", "name"}),
		traceTokens:     prometheus.NewCounterVec(prometheus.CounterOpts{Name: "llm_obs_trace_tokens_total", Help: "Tokens seen in traces."}, []string{"runtime", "model", "name", "kind"}),
		traceTPS:        prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_trace_tokens_per_sec", Help: "Throughput of the most recent trace."}, []string{"runtime", "model", "name"}),
		traceConcurrent: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "llm_obs_trace_concurrent", Help: "Concurrent traces observed by the most recent trace."}, []string{"runtime", "model", "name"}),
	}
	reg.MustRegister(e.tokensPerSec, e.memUsed, e.memTotal, e.modelMem, e.modelGPUMem, e.kvCache, e.ctxMax, e.reqRunning, e.reqWaiting,
		e.modelLoaded, e.events, e.traces, e.traceDuration, e.traceTokens, e.traceTPS, e.traceConcurrent)
	return e
}

// Handler serves /metrics.
func (e *Exporter) Handler() http.Handler {
	return promhttp.HandlerFor(e.reg, promhttp.HandlerOpts{})
}

// Registry exposes the registry (tests).
func (e *Exporter) Registry() *prometheus.Registry { return e.reg }

// ObserveSnapshot updates gauges from a snapshot.
func (e *Exporter) ObserveSnapshot(s core.RuntimeSnapshot) {
	dev := s.BackendDevice
	if dev == "" {
		dev = "unknown"
	}
	rmd := prometheus.Labels{"runtime": s.Runtime, "model": s.Model, "device": dev}
	rm := prometheus.Labels{"runtime": s.Runtime, "model": s.Model}

	e.mu.Lock()
	if prev, ok := e.labels[s.Key()]; ok && prev["device"] != dev {
		// Device changed: drop the stale series so dashboards don't show both.
		e.tokensPerSec.Delete(prev)
		e.modelMem.Delete(prev)
		e.modelGPUMem.Delete(prev)
		e.modelLoaded.Delete(prev)
	}
	e.labels[s.Key()] = rmd
	e.mu.Unlock()

	e.tokensPerSec.With(rmd).Set(s.TokensPerSec)
	e.modelMem.With(rmd).Set(float64(s.ModelMemMB))
	e.modelGPUMem.With(rmd).Set(float64(s.ModelGPUMemMB))
	e.modelLoaded.With(rmd).Set(1)
	e.memUsed.WithLabelValues(s.Runtime).Set(float64(s.MemUsedMB))
	e.memTotal.WithLabelValues(s.Runtime).Set(float64(s.MemTotalMB))
	e.kvCache.With(rm).Set(s.KVCacheUsedPct)
	e.ctxMax.With(rm).Set(float64(s.ContextMaxTok))
	e.reqRunning.With(rm).Set(float64(s.RequestsRunning))
	e.reqWaiting.With(rm).Set(float64(s.RequestsWaiting))
}

// ObserveEvent counts events and clears gauges for unloaded models.
func (e *Exporter) ObserveEvent(ev core.Event) {
	e.events.WithLabelValues(ev.Runtime, string(ev.Kind)).Inc()
	if ev.Kind == core.EventModelUnloaded || ev.Kind == core.EventRuntimeDown {
		key := ev.Runtime + "/" + ev.Model
		e.mu.Lock()
		for k, lbl := range e.labels {
			if k == key || (ev.Kind == core.EventRuntimeDown && lbl["runtime"] == ev.Runtime) {
				e.tokensPerSec.Delete(lbl)
				e.modelMem.Delete(lbl)
				e.modelGPUMem.Delete(lbl)
				e.modelLoaded.Delete(lbl)
				rm := prometheus.Labels{"runtime": lbl["runtime"], "model": lbl["model"]}
				e.kvCache.Delete(rm)
				e.ctxMax.Delete(rm)
				e.reqRunning.Delete(rm)
				e.reqWaiting.Delete(rm)
				delete(e.labels, k)
			}
		}
		e.mu.Unlock()
	}
}

// ObserveTrace records a completed trace.
func (e *Exporter) ObserveTrace(t trace.Trace) {
	status := "ok"
	if t.Error != "" {
		status = "error"
	}
	e.traces.WithLabelValues(t.Runtime, t.Model, t.Name, status).Inc()
	e.traceDuration.WithLabelValues(t.Runtime, t.Model, t.Name).Observe(t.DurationMS / 1000)
	e.traceTokens.WithLabelValues(t.Runtime, t.Model, t.Name, "prompt").Add(float64(t.PromptTokens))
	e.traceTokens.WithLabelValues(t.Runtime, t.Model, t.Name, "completion").Add(float64(t.CompletionTokens))
	if t.TokensPerSec > 0 {
		e.traceTPS.WithLabelValues(t.Runtime, t.Model, t.Name).Set(t.TokensPerSec)
	}
	e.traceConcurrent.WithLabelValues(t.Runtime, t.Model, t.Name).Set(float64(t.Context.ConcurrentTraces))
}
