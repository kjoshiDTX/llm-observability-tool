package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/hostmem"
)

// VLLMAdapter scrapes vLLM's native Prometheus /metrics endpoint.
//
// Metric names (vLLM >= 0.6, V1 engine): all prefixed "vllm:".
//
//	num_requests_running / num_requests_waiting   gauges, label model_name
//	gpu_cache_usage_perc (V0) / kv_cache_usage_perc (V1)   gauge 0..1
//	generation_tokens_total / prompt_tokens_total counters -> rate = tokens/sec
//	time_per_output_token_seconds                 histogram (fallback tps)
//
// vLLM does not report which device it runs on. On Apple Silicon there is no
// Metal backend, so vLLM is CPU-only; we default DefaultDevice to "cpu" on
// darwin and "cuda" elsewhere, and any GPU-specific metric that is missing
// simply leaves its field at zero rather than failing the poll.
type VLLMAdapter struct {
	MetricsURL    string // e.g. http://localhost:8000/metrics
	DefaultDevice string
	Client        *http.Client

	mu   sync.Mutex
	prev map[string]counterSample // model -> last generation_tokens_total
	ctx  map[string]int           // model -> max_model_len from /v1/models
}

type counterSample struct {
	value float64
	at    time.Time
}

// NewVLLMAdapter builds an adapter for the given metrics URL.
func NewVLLMAdapter(metricsURL, defaultDevice string) *VLLMAdapter {
	if metricsURL == "" {
		metricsURL = "http://localhost:8000/metrics"
	}
	if defaultDevice == "" {
		if hostmem.Unified() {
			defaultDevice = "cpu" // no Metal backend in vLLM
		} else {
			defaultDevice = "cuda"
		}
	}
	return &VLLMAdapter{
		MetricsURL:    metricsURL,
		DefaultDevice: defaultDevice,
		Client:        &http.Client{Timeout: 3 * time.Second},
		prev:          map[string]counterSample{},
		ctx:           map[string]int{},
	}
}

func (a *VLLMAdapter) Name() string { return "vllm" }

// Poll implements core.Adapter.
func (a *VLLMAdapter) Poll(ctx context.Context) ([]core.RuntimeSnapshot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.MetricsURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := a.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: vllm: %v", core.ErrRuntimeUnavailable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: vllm: status %d", core.ErrRuntimeUnavailable, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("vllm: read metrics: %w", err)
	}
	snaps, err := a.ParseMetrics(body, time.Now())
	if err != nil {
		return nil, err
	}
	// Best-effort: fill context max from the OpenAI-compatible /v1/models.
	for i := range snaps {
		if snaps[i].ContextMaxTok == 0 {
			snaps[i].ContextMaxTok = a.maxModelLen(ctx, snaps[i].Model)
		}
	}
	return snaps, nil
}

// ParseMetrics converts a Prometheus text exposition into snapshots. It is
// deliberately tolerant: a malformed body returns an error (never panics),
// and missing metrics leave fields at zero.
func (a *VLLMAdapter) ParseMetrics(body []byte, now time.Time) ([]core.RuntimeSnapshot, error) {
	parser := expfmt.NewTextParser(model.UTF8Validation)
	families, err := parser.TextToMetricFamilies(strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("vllm: parse metrics: %w", err)
	}
	host, _ := hostmem.Read()

	type acc struct {
		snap      core.RuntimeSnapshot
		genTokens float64
		haveGen   bool
		tpotSum   float64
		tpotCnt   float64
	}
	byModel := map[string]*acc{}
	get := func(model string) *acc {
		x, ok := byModel[model]
		if !ok {
			x = &acc{snap: core.RuntimeSnapshot{
				Runtime:       "vllm",
				Model:         model,
				MemUsedMB:     host.UsedMB,
				MemTotalMB:    host.TotalMB,
				UnifiedMemory: host.Unified,
				BackendDevice: a.DefaultDevice,
				Timestamp:     now,
			}}
			byModel[model] = x
		}
		return x
	}

	for name, mf := range families {
		base := strings.TrimPrefix(name, "vllm:")
		for _, m := range mf.GetMetric() {
			model := labelValue(m, "model_name")
			if model == "" {
				continue
			}
			x := get(model)
			switch base {
			case "num_requests_running":
				x.snap.RequestsRunning = int(gaugeOrUntyped(m))
			case "num_requests_waiting":
				x.snap.RequestsWaiting = int(gaugeOrUntyped(m))
			case "gpu_cache_usage_perc", "kv_cache_usage_perc":
				v := gaugeOrUntyped(m)
				if v <= 1.0 {
					v *= 100
				}
				x.snap.KVCacheUsedPct = v
			case "generation_tokens_total":
				if c := m.GetCounter(); c != nil {
					x.genTokens, x.haveGen = c.GetValue(), true
				} else if u := m.GetUntyped(); u != nil {
					x.genTokens, x.haveGen = u.GetValue(), true
				}
			case "time_per_output_token_seconds":
				if h := m.GetHistogram(); h != nil {
					x.tpotSum, x.tpotCnt = h.GetSampleSum(), float64(h.GetSampleCount())
				}
			}
		}
	}

	out := make([]core.RuntimeSnapshot, 0, len(byModel))
	a.mu.Lock()
	defer a.mu.Unlock()
	for model, x := range byModel {
		// tokens/sec from the counter rate between this and the previous poll.
		if x.haveGen {
			if p, ok := a.prev[model]; ok {
				dt := now.Sub(p.at).Seconds()
				if dt > 0 && x.genTokens >= p.value {
					x.snap.TokensPerSec = (x.genTokens - p.value) / dt
					x.snap.TPSSource = "runtime"
				}
			}
			a.prev[model] = counterSample{value: x.genTokens, at: now}
		}
		// Fallback: 1 / mean time-per-output-token (lifetime average).
		if x.snap.TokensPerSec == 0 && x.tpotCnt > 0 && x.tpotSum > 0 && x.snap.RequestsRunning > 0 {
			x.snap.TokensPerSec = x.tpotCnt / x.tpotSum
			x.snap.TPSSource = "runtime"
		}
		out = append(out, x.snap)
	}
	return out, nil
}

func (a *VLLMAdapter) maxModelLen(ctx context.Context, model string) int {
	a.mu.Lock()
	if n, ok := a.ctx[model]; ok {
		a.mu.Unlock()
		return n
	}
	a.mu.Unlock()
	base := strings.TrimSuffix(a.MetricsURL, "/metrics")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return 0
	}
	resp, err := a.Client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var r struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int    `json:"max_model_len"`
		} `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&r) != nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, d := range r.Data {
		a.ctx[d.ID] = d.MaxModelLen
	}
	return a.ctx[model]
}

func labelValue(m *dto.Metric, name string) string {
	for _, l := range m.GetLabel() {
		if l.GetName() == name {
			return l.GetValue()
		}
	}
	return ""
}

func gaugeOrUntyped(m *dto.Metric) float64 {
	if g := m.GetGauge(); g != nil {
		return g.GetValue()
	}
	if u := m.GetUntyped(); u != nil {
		return u.GetValue()
	}
	if c := m.GetCounter(); c != nil {
		return c.GetValue()
	}
	return 0
}
