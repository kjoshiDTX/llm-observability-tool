package adapters

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

const psFixture = `{"models":[{"name":"qwen2.5:0.5b","model":"qwen2.5:0.5b","size":1491231232,"digest":"abc","details":{"parameter_size":"494.03M","quantization_level":"Q4_K_M","family":"qwen2"},"expires_at":"2026-09-18T17:02:13.075217-05:00","size_vram":1491231232}]}`

func TestOllamaPollParsesPS(t *testing.T) {
	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			_, _ = w.Write([]byte(psFixture))
		case "/api/show":
			showCalls.Add(1)
			_, _ = w.Write([]byte(`{"details":{"parameter_size":"494.03M","quantization_level":"Q4_K_M"},"model_info":{"qwen2.context_length":32768,"qwen2.embedding_length":896,"qwen2.block_count":24,"qwen2.attention.head_count":14,"qwen2.attention.head_count_kv":2}}`))
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:0.5b","size":397821319}]}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()
	a := NewOllamaAdapter(srv.URL)
	snaps, err := a.Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snaps) != 1 {
		t.Fatalf("want 1 snapshot, got %d", len(snaps))
	}
	s := snaps[0]
	if s.Model != "qwen2.5:0.5b" || s.ModelMemMB != 1422 || s.ModelGPUMemMB != 1422 {
		t.Errorf("residency wrong: %+v", s)
	}
	if s.BackendDevice != "mps" && s.BackendDevice != "cuda" {
		t.Errorf("device = %q", s.BackendDevice)
	}
	if s.ContextMaxTok != 32768 || s.Quantization != "Q4_K_M" {
		t.Errorf("metadata wrong: %+v", s)
	}
	// Second poll must use the cached /api/show.
	if _, err := a.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if showCalls.Load() != 1 {
		t.Errorf("show called %d times, want 1 (cached)", showCalls.Load())
	}
	info, err := a.Show(context.Background(), "qwen2.5:0.5b")
	if err != nil || info.BlockCount != 24 || info.HeadCountKV != 2 || info.FileSizeBytes != 397821319 {
		t.Errorf("show: %+v %v", info, err)
	}
}

func TestOllamaNotRunning(t *testing.T) {
	a := NewOllamaAdapter("http://127.0.0.1:1")
	snaps, err := a.Poll(context.Background())
	if !isUnavailable(err) {
		t.Fatalf("want ErrRuntimeUnavailable, got %v", err)
	}
	if len(snaps) != 0 {
		t.Fatalf("want empty slice")
	}
}

func TestBackendDevice(t *testing.T) {
	if backendDevice(100, 0) != "cpu" {
		t.Error("all-CPU should be cpu")
	}
	if backendDevice(100, 50) != "mixed" {
		t.Error("partial offload should be mixed")
	}
	if d := backendDevice(100, 100); d != "mps" && d != "cuda" {
		t.Errorf("full GPU = %q", d)
	}
	if backendDevice(0, 0) != "unknown" {
		t.Error("zero size should be unknown")
	}
}
