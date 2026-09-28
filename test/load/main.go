// Command load drives a running llm-obs collector with a synthetic stream of
// traces for a sustained period and measures the collector process's own
// CPU/memory footprint, per Phase 9 of the build plan. It does not replace
// the real end-to-end verification against live Ollama (see
// examples/simulate_calls.py and RESULTS.md) -- it exists to answer a
// different question: does the collector itself stay cheap under sustained
// concurrent load, independent of how slow the underlying LLM runtime is.
//
// Usage:
//
//	go build -o bin/llm-obs-collector ./collector/cmd/collector
//	./bin/llm-obs-collector -db /tmp/load.db -listen :8765 &
//	go run ./test/load -duration 10m -rate 20 -concurrency 8 -pid $(pgrep -f llm-obs-collector | head -1)
//
// Or let it manage the collector subprocess itself with -spawn (default),
// which also makes -pid unnecessary.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type sample struct {
	t      time.Time
	cpuPct float64
	rssKB  int64
	numGor int
}

func main() {
	url := flag.String("url", "http://localhost:8765", "collector base URL")
	duration := flag.Duration("duration", 10*time.Minute, "how long to run")
	rate := flag.Float64("rate", 20, "target traces/sec")
	concurrency := flag.Int("concurrency", 8, "concurrent trace senders")
	names := flag.Int("names", 5, "distinct trace names (endpoints) to simulate")
	models := flag.Int("models", 2, "distinct model names to simulate")
	pid := flag.Int("pid", 0, "PID of an already-running collector to sample (skips -spawn)")
	spawn := flag.Bool("spawn", true, "spawn the collector binary if -pid is not given")
	binPath := flag.String("bin", "./bin/llm-obs-collector", "collector binary path (with -spawn)")
	dbPath := flag.String("db", "", "db path for the spawned collector (default: a temp file)")
	resultsPath := flag.String("out", "RESULTS.md", "where to append the load-test report")
	sampleEvery := flag.Duration("sample-every", time.Second, "process-sample interval")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *duration+30*time.Second)
	defer cancel()

	targetPID := *pid
	var cmd *exec.Cmd
	if targetPID == 0 {
		if !*spawn {
			log.Fatal("must pass -pid or leave -spawn=true")
		}
		db := *dbPath
		if db == "" {
			f, err := os.CreateTemp("", "llm-obs-load-*.db")
			if err != nil {
				log.Fatal(err)
			}
			f.Close()
			db = f.Name()
		}
		cmd = exec.CommandContext(ctx, *binPath, "-db", db, "-listen", mustPort(*url), "-ollama", "", "-vllm", "")
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		if err := cmd.Start(); err != nil {
			log.Fatalf("spawn collector: %v", err)
		}
		targetPID = cmd.Process.Pid
		log.Printf("spawned collector pid=%d db=%s", targetPID, db)
		time.Sleep(500 * time.Millisecond)
	}

	// ---- health check ----
	if err := waitHealthy(*url, 10*time.Second); err != nil {
		log.Fatalf("collector not healthy at %s: %v", *url, err)
	}

	// ---- CPU/RSS sampler ----
	var samples []sample
	var sampleMu sync.Mutex
	sampleDone := make(chan struct{})
	go func() {
		defer close(sampleDone)
		t := time.NewTicker(*sampleEvery)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				cpuPct, rssKB, err := psSample(targetPID)
				if err != nil {
					continue
				}
				var numGor int
				if r := httpGetHealth(*url); r != nil {
					numGor = r.Runtimes
				}
				sampleMu.Lock()
				samples = append(samples, sample{time.Now(), cpuPct, rssKB, numGor})
				sampleMu.Unlock()
			}
		}
	}()

	// ---- host baseline (for "share of the machine" context) ----
	hostCPUs := runtime.NumCPU()

	// ---- load generator ----
	var sent, sendErrs, begun int64
	log.Printf("generating load: ~%.1f traces/sec, %d senders, for %s", *rate, *concurrency, *duration)
	genCtx, genCancel := context.WithTimeout(ctx, *duration)
	defer genCancel()

	perSender := *rate / float64(*concurrency)
	interval := time.Duration(float64(time.Second) / perSender)
	var wg sync.WaitGroup
	client := &http.Client{Timeout: 5 * time.Second}
	for w := 0; w < *concurrency; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(worker) + time.Now().UnixNano()))
			t := time.NewTicker(interval)
			defer t.Stop()
			for {
				select {
				case <-genCtx.Done():
					return
				case <-t.C:
					name := fmt.Sprintf("endpoint_%d", rng.Intn(*names))
					model := fmt.Sprintf("model_%d", rng.Intn(*models))
					id := fmt.Sprintf("%d-%d-%d", worker, time.Now().UnixNano(), rng.Int63())
					start := time.Now()
					atomic.AddInt64(&begun, 1)
					postJSON(client, *url+"/traces/begin", map[string]any{
						"id": id, "name": name, "model": model, "runtime": "ollama", "start_time": start,
					})
					// Simulate realistic-ish call latency: mostly fast, occasional slow tail.
					simMS := 50 + rng.Intn(150)
					if rng.Intn(20) == 0 {
						simMS += rng.Intn(2000) // ~5% slow tail
					}
					time.Sleep(time.Duration(simMS) * time.Millisecond)
					end := time.Now()
					tokens := 10 + rng.Intn(100)
					status, err := postJSON(client, *url+"/traces", map[string]any{
						"id": id, "name": name, "model": model, "runtime": "ollama",
						"start_time": start, "end_time": end,
						"completion_tokens": tokens, "tokens_per_sec": float64(tokens) / (float64(simMS) / 1000),
						"tags": map[string]string{"load_test": "true"},
					})
					if err != nil || status >= 300 {
						atomic.AddInt64(&sendErrs, 1)
					} else {
						atomic.AddInt64(&sent, 1)
					}
				}
			}
		}(w)
	}
	wg.Wait()
	<-sampleDone

	// ---- report ----
	sampleMu.Lock()
	defer sampleMu.Unlock()
	report := buildReport(reportInput{
		duration: *duration, targetRate: *rate, concurrency: *concurrency, hostCPUs: hostCPUs,
		begun: begun, sent: sent, sendErrs: sendErrs, samples: samples, url: *url,
	})
	fmt.Println(report)
	if *resultsPath != "" {
		appendToFile(*resultsPath, report)
		log.Printf("report appended to %s", *resultsPath)
	}

	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}

type healthResp struct {
	Runtimes int
}

func httpGetHealth(base string) *healthResp {
	resp, err := http.Get(base + "/health")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var raw struct {
		Runtimes []any `json:"runtimes"`
	}
	if json.NewDecoder(resp.Body).Decode(&raw) != nil {
		return nil
	}
	return &healthResp{Runtimes: len(raw.Runtimes)}
}

func waitHealthy(base string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		resp, err := http.Get(base + "/health")
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			return nil
		}
		if err != nil {
			lastErr = err
		}
		time.Sleep(200 * time.Millisecond)
	}
	return lastErr
}

func postJSON(client *http.Client, url string, body any) (int, error) {
	b, _ := json.Marshal(body)
	resp, err := client.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}

func appendToFile(path, content string) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("could not append to %s: %v", path, err)
		return
	}
	defer f.Close()
	fmt.Fprintln(f, content)
}

func mustPort(url string) string {
	// crude: extract ":PORT" suffix from a URL like http://localhost:8765
	for i := len(url) - 1; i >= 0; i-- {
		if url[i] == ':' {
			return url[i:]
		}
	}
	return ":8765"
}

type reportInput struct {
	duration    time.Duration
	targetRate  float64
	concurrency int
	hostCPUs    int
	begun       int64
	sent        int64
	sendErrs    int64
	samples     []sample
	url         string
}

func buildReport(in reportInput) string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "## Load test: %s\n\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "- Duration: %s, target rate: %.1f traces/sec, %d concurrent senders\n", in.duration, in.targetRate, in.concurrency)
	fmt.Fprintf(&b, "- Traces begun: %d, completed successfully: %d, send errors: %d (%.2f%%)\n", in.begun, in.sent, in.sendErrs, pct(in.sendErrs, in.begun))
	if len(in.samples) == 0 {
		fmt.Fprintln(&b, "- No process samples were collected (is `ps` available and the PID valid?).")
		return b.String()
	}
	var cpuSum, cpuMax float64
	var rssSum, rssMax int64
	for _, s := range in.samples {
		cpuSum += s.cpuPct
		if s.cpuPct > cpuMax {
			cpuMax = s.cpuPct
		}
		rssSum += s.rssKB
		if s.rssKB > rssMax {
			rssMax = s.rssKB
		}
	}
	n := float64(len(in.samples))
	avgCPU := cpuSum / n
	avgRSS := rssSum / int64(n)
	avgCPUShareOfMachine := avgCPU / float64(in.hostCPUs*100) * 100

	fmt.Fprintf(&b, "- Collector CPU: mean %.2f%% of one core (%.3f%% of this %d-core machine), max %.2f%% of one core\n",
		avgCPU, avgCPUShareOfMachine, in.hostCPUs, cpuMax)
	fmt.Fprintf(&b, "- Collector RSS: mean %.1f MB, max %.1f MB\n", float64(avgRSS)/1024, float64(rssMax)/1024)
	fmt.Fprintf(&b, "- %d process samples over %s\n", len(in.samples), in.duration)

	// The plan's "<1-2% overhead" target means overhead as a share of the
	// machine's total capacity, not as a share of one core: a background
	// collector pegging 9% of a single core on a 10-core box is cheap (under
	// 1% of the machine); the same 9% on a 1-core box would not be. Compare
	// against the machine-wide share, and report the raw one-core figure
	// alongside it purely for transparency.
	verdict := "within target (<1-2% of this machine's total CPU capacity on average)"
	if avgCPUShareOfMachine >= 2.0 {
		verdict = "ABOVE the <1-2% of total machine capacity target -- investigate before calling this production-ready"
	}
	fmt.Fprintf(&b, "- **Verdict: %s.**\n", verdict)
	return b.String()
}

func pct(n, d int64) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d) * 100
}
