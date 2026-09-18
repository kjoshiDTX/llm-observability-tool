package adapters

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
)

const vllmFixture = `# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.42
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1000.0
# HELP vllm:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.0
vllm:time_per_output_token_seconds_bucket{le="0.05",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 900.0
vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1000.0
vllm:time_per_output_token_seconds_sum{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 25.0
vllm:time_per_output_token_seconds_count{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1000.0
`

func TestVLLMParseFixture(t *testing.T) {
	a := NewVLLMAdapter("http://x/metrics", "cpu")
	t0 := time.Now()
	snaps, err := a.ParseMetrics([]byte(vllmFixture), t0)
	if err != nil {
		t.Fatal(err)
	}
	if len(snaps) != 1 {
		t.Fatalf("want 1 snapshot, got %d", len(snaps))
	}
	s := snaps[0]
	if s.Model != "Qwen/Qwen2.5-0.5B-Instruct" || s.Runtime != "vllm" {
		t.Errorf("identity wrong: %+v", s)
	}
	if s.RequestsRunning != 2 || s.RequestsWaiting != 1 {
		t.Errorf("concurrency wrong: %+v", s)
	}
	if s.KVCacheUsedPct != 42 {
		t.Errorf("kv cache pct = %v, want 42", s.KVCacheUsedPct)
	}
	if s.BackendDevice != "cpu" {
		t.Errorf("device = %q", s.BackendDevice)
	}
	// First poll: no previous counter, so tps falls back to 1/mean TPOT = 1000/25 = 40.
	if s.TokensPerSec != 40 {
		t.Errorf("fallback tps = %v, want 40", s.TokensPerSec)
	}

	// Second poll two seconds later with +100 tokens => 50 tok/s from counter rate.
	body2 := strings.Replace(vllmFixture, `generation_tokens_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1000.0`,
		`generation_tokens_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1100.0`, 1)
	snaps2, err := a.ParseMetrics([]byte(body2), t0.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if got := snaps2[0].TokensPerSec; got != 50 {
		t.Errorf("counter-rate tps = %v, want 50", got)
	}
}

// V1-engine rename: kv_cache_usage_perc replaces gpu_cache_usage_perc.
func TestVLLMParseV1KVCacheName(t *testing.T) {
	body := "vllm:kv_cache_usage_perc{model_name=\"m\"} 0.9\nvllm:num_requests_running{model_name=\"m\"} 0\n"
	a := NewVLLMAdapter("", "")
	snaps, err := a.ParseMetrics([]byte(body), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if snaps[0].KVCacheUsedPct != 90 {
		t.Errorf("kv = %v", snaps[0].KVCacheUsedPct)
	}
}

// CPU mode: GPU-specific metrics absent entirely. Must not error.
func TestVLLMMissingGPUMetricsIsGraceful(t *testing.T) {
	body := "vllm:num_requests_running{model_name=\"m\"} 1\n"
	a := NewVLLMAdapter("", "cpu")
	snaps, err := a.ParseMetrics([]byte(body), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(snaps) != 1 || snaps[0].KVCacheUsedPct != 0 || snaps[0].BackendDevice != "cpu" {
		t.Errorf("unexpected: %+v", snaps)
	}
}

// Chaos: malformed exposition text must produce an error, never a panic, and
// garbage that happens to parse must not produce bogus models.
func TestVLLMMalformedMetrics(t *testing.T) {
	cases := []string{
		"vllm:num_requests_running{model_name=\"m\" 1\n",    // unclosed label brace
		"vllm:num_requests_running{model_name=\"m\"} abc\n", // non-numeric value
		"# TYPE vllm:x gauge\nvllm:x{model_name=\"m\"}\n",   // missing value
		"\x00\xff\xfe binary garbage",
		strings.Repeat("a", 100000),
	}
	a := NewVLLMAdapter("", "cpu")
	for i, body := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("case %d panicked: %v", i, r)
				}
			}()
			snaps, err := a.ParseMetrics([]byte(body), time.Now())
			if err == nil && len(snaps) > 0 {
				t.Errorf("case %d: expected error or empty, got %+v", i, snaps)
			}
		}()
	}
	// Empty body is valid: runtime up, nothing to report.
	if snaps, err := a.ParseMetrics(nil, time.Now()); err != nil || len(snaps) != 0 {
		t.Errorf("empty body: %v %v", snaps, err)
	}
}

func TestVLLMPollUnavailable(t *testing.T) {
	a := NewVLLMAdapter("http://127.0.0.1:1/metrics", "cpu")
	_, err := a.Poll(context.Background())
	if err == nil || !isUnavailable(err) {
		t.Fatalf("want ErrRuntimeUnavailable, got %v", err)
	}
}

func TestVLLMPollAgainstServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			_, _ = w.Write([]byte(vllmFixture))
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"Qwen/Qwen2.5-0.5B-Instruct","max_model_len":32768}]}`))
		}
	}))
	defer srv.Close()
	a := NewVLLMAdapter(srv.URL+"/metrics", "cpu")
	snaps, err := a.Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snaps[0].ContextMaxTok != 32768 {
		t.Errorf("max_model_len not filled: %+v", snaps[0])
	}
}

func isUnavailable(err error) bool {
	for e := err; e != nil; {
		if e == core.ErrRuntimeUnavailable {
			return true
		}
		u, ok := e.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		e = u.Unwrap()
	}
	return false
}
