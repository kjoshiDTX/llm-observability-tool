// Command fakevllm serves a vLLM-shaped Prometheus /metrics endpoint (and
// /v1/models) with slowly evolving values, so the collector's vLLM adapter
// can be exercised on machines where vLLM cannot run (Apple Silicon has no
// Metal backend; vLLM there is CPU-only and slow to set up).
//
// Chaos knob: -malformed serves garbage instead of metrics.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync/atomic"
	"time"
)

func main() {
	addr := flag.String("addr", ":8000", "listen address")
	model := flag.String("model", "Qwen/Qwen2.5-0.5B-Instruct", "model name label")
	malformed := flag.Bool("malformed", false, "serve malformed exposition text")
	kv := flag.Float64("kv", 0.35, "baseline kv cache usage (0..1)")
	flag.Parse()

	start := time.Now()
	var tokens atomic.Int64
	go func() {
		for range time.Tick(200 * time.Millisecond) {
			// ~45 tok/s with a slow sinusoidal wobble.
			tokens.Add(int64(9 + 3*math.Sin(time.Since(start).Seconds()/10)))
		}
	}()

	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if *malformed {
			_, _ = w.Write([]byte("vllm:num_requests_running{model_name=\"x\" 1\n\x00garbage"))
			return
		}
		t := time.Since(start).Seconds()
		running := 1 + int(2*math.Abs(math.Sin(t/7)))
		waiting := int(math.Abs(math.Sin(t/13)) * 2)
		kvNow := *kv + 0.3*math.Abs(math.Sin(t/20))
		n := tokens.Load()
		fmt.Fprintf(w, `# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name=%q} %d
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name=%q} %d
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name=%q} %.3f
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name=%q} %d
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name=%q} %d
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.025",model_name=%q} %d
vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name=%q} %d
vllm:time_per_output_token_seconds_sum{model_name=%q} %.3f
vllm:time_per_output_token_seconds_count{model_name=%q} %d
`, *model, running, *model, waiting, *model, kvNow, *model, n, *model, n/3,
			*model, n*9/10, *model, n, *model, float64(n)/45.0, *model, n)
	})
	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"object":"model","max_model_len":32768}]}`, *model)
	})
	log.Printf("fake vLLM metrics on %s (model=%s malformed=%v)", *addr, *model, *malformed)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
