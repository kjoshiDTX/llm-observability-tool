// Command collector runs the llm-obs collector daemon: it polls runtimes,
// stores snapshots/traces in SQLite, and serves the local HTTP API.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/adapters"
	"github.com/krishjoshi/llm-obs/collector/internal/api"
	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/core/diagnose"
	"github.com/krishjoshi/llm-obs/collector/internal/core/preflight"
	"github.com/krishjoshi/llm-obs/collector/internal/export"
	"github.com/krishjoshi/llm-obs/collector/internal/storage"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

const version = "0.1.0"

func main() {
	listen := flag.String("listen", envOr("LLM_OBS_LISTEN", ":8765"), "HTTP listen address")
	dbPath := flag.String("db", envOr("LLM_OBS_DB", "llm-obs.db"), "SQLite database path")
	ollamaURL := flag.String("ollama", envOr("LLM_OBS_OLLAMA_URL", "http://localhost:11434"), "Ollama base URL ('' to disable)")
	vllmURL := flag.String("vllm", envOr("LLM_OBS_VLLM_METRICS_URL", "http://localhost:8000/metrics"), "vLLM /metrics URL ('' to disable)")
	vllmDevice := flag.String("vllm-device", envOr("LLM_OBS_VLLM_DEVICE", ""), "device label for vLLM (default: cpu on Apple Silicon, cuda elsewhere)")
	interval := flag.Duration("interval", 2*time.Second, "poll interval")
	retention := flag.Duration("retention", 24*time.Hour, "how long to keep snapshots/events")
	printSnaps := flag.Bool("print", false, "print every snapshot as JSON to stdout")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	store, err := storage.Open(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	store.StartWriter(ctx)

	c := core.NewCollector()
	if *ollamaURL != "" {
		c.Register(adapters.NewOllamaAdapter(*ollamaURL), *interval)
	}
	if *vllmURL != "" {
		c.Register(adapters.NewVLLMAdapter(*vllmURL, *vllmDevice), *interval)
	}
	exporter := export.New()
	snapHook := store.SnapshotHook()
	enc := json.NewEncoder(os.Stdout)
	c.OnSnapshot = func(s core.RuntimeSnapshot) {
		snapHook(s)
		exporter.ObserveSnapshot(s)
		if *printSnaps {
			_ = enc.Encode(s)
		}
	}
	c.OnEvent = func(e core.Event) {
		_ = store.InsertEvent(e)
		exporter.ObserveEvent(e)
	}

	ingester := trace.NewIngester(store, c)
	diagnoser := diagnose.New(store, c)
	srv := &api.Server{
		Collector: c,
		Store:     store,
		Ingester:  ingester,
		Diagnoser: diagnoser,
		Comparer:  diagnose.NewComparer(store, diagnoser),
		Checker:   preflight.New(c, ollamaShow(*ollamaURL)),
		Metrics:   exporter.Handler(),
		Logger:    log.Default(),
		Version:   version,
	}
	ingester.OnIngest = exporter.ObserveTrace

	httpSrv := &http.Server{Addr: *listen, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		log.Printf("llm-obs collector %s listening on %s (db=%s ollama=%q vllm=%q)", version, *listen, *dbPath, *ollamaURL, *vllmURL)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http: %v", err)
		}
	}()

	// Periodic retention pruning.
	go func() {
		t := time.NewTicker(10 * time.Minute)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if err := store.Prune(*retention); err != nil {
					log.Printf("prune: %v", err)
				}
			}
		}
	}()

	c.Run(ctx) // blocks until SIGINT/SIGTERM

	log.Printf("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdownCtx)
	if err := store.Close(); err != nil {
		log.Printf("close db: %v", err)
	}
	log.Printf("collector stopped")
}

func ollamaShow(url string) preflight.ModelInfoFunc {
	if url == "" {
		return nil
	}
	ad := adapters.NewOllamaAdapter(url)
	return func(ctx context.Context, model string) (preflight.ModelInfo, error) {
		info, err := ad.Show(ctx, model)
		return preflight.ModelInfo{
			AvailableLocal: info.AvailableLocal,
			ParameterSize:  info.ParameterSize,
			Quantization:   info.Quantization,
			ContextLength:  info.ContextLength,
			EmbeddingDim:   info.EmbeddingDim,
			BlockCount:     info.BlockCount,
			HeadCount:      info.HeadCount,
			HeadCountKV:    info.HeadCountKV,
			FileSizeBytes:  info.FileSizeBytes,
		}, err
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
