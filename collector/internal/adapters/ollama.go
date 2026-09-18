// Package adapters contains one Adapter per supported runtime.
package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/hostmem"
)

// OllamaAdapter polls Ollama's local HTTP API. Ollama exposes no metrics
// endpoint, so throughput and context usage are filled in later from traces;
// what we get here is residency (which models are loaded, how much memory,
// how much of it is on the accelerator) and model metadata.
type OllamaAdapter struct {
	BaseURL string
	Client  *http.Client

	mu       sync.Mutex
	showInfo map[string]ShowInfo // per-model metadata cache (context length etc.)
}

// NewOllamaAdapter returns an adapter for the given base URL (default
// http://localhost:11434).
func NewOllamaAdapter(baseURL string) *OllamaAdapter {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &OllamaAdapter{
		BaseURL:  baseURL,
		Client:   &http.Client{Timeout: 3 * time.Second},
		showInfo: map[string]ShowInfo{},
	}
}

func (a *OllamaAdapter) Name() string { return "ollama" }

// psResponse mirrors GET /api/ps.
type psResponse struct {
	Models []struct {
		Name      string    `json:"name"`
		Model     string    `json:"model"`
		Size      int64     `json:"size"`
		SizeVRAM  int64     `json:"size_vram"`
		ExpiresAt time.Time `json:"expires_at"`
		Details   struct {
			ParameterSize     string `json:"parameter_size"`
			QuantizationLevel string `json:"quantization_level"`
			Family            string `json:"family"`
		} `json:"details"`
		ContextLength int `json:"context_length"`
	} `json:"models"`
}

// Poll implements core.Adapter.
func (a *OllamaAdapter) Poll(ctx context.Context) ([]core.RuntimeSnapshot, error) {
	var ps psResponse
	if err := a.getJSON(ctx, "/api/ps", &ps); err != nil {
		return nil, fmt.Errorf("%w: ollama: %v", core.ErrRuntimeUnavailable, err)
	}
	now := time.Now()
	host, _ := hostmem.Read() // best effort; zero values are tolerated

	out := make([]core.RuntimeSnapshot, 0, len(ps.Models))
	for _, m := range ps.Models {
		const mb = 1024 * 1024
		snap := core.RuntimeSnapshot{
			Runtime:       "ollama",
			Model:         m.Name,
			MemUsedMB:     host.UsedMB,
			MemTotalMB:    host.TotalMB,
			UnifiedMemory: host.Unified,
			ModelMemMB:    m.Size / mb,
			ModelGPUMemMB: m.SizeVRAM / mb,
			ContextMaxTok: a.contextLength(ctx, m.Name, m.ContextLength),
			BackendDevice: backendDevice(m.Size, m.SizeVRAM),
			ParameterSize: m.Details.ParameterSize,
			Quantization:  m.Details.QuantizationLevel,
			Timestamp:     now,
		}
		out = append(out, snap)
	}
	return out, nil
}

// contextLength prefers the loaded context (/api/ps, Ollama >= 0.8), falling
// back to the model's architectural maximum from /api/show. The fallback is
// an upper bound: Ollama's default num_ctx is smaller than the model max.
func (a *OllamaAdapter) contextLength(ctx context.Context, model string, fromPS int) int {
	if fromPS > 0 {
		return fromPS
	}
	a.mu.Lock()
	info, ok := a.showInfo[model]
	a.mu.Unlock()
	if !ok {
		var err error
		info, err = a.Show(ctx, model)
		if err != nil {
			return 0
		}
		a.mu.Lock()
		a.showInfo[model] = info
		a.mu.Unlock()
	}
	return info.ContextLength
}

// backendDevice classifies where a model is executing from Ollama's
// size / size_vram split. On Apple Silicon "vram" means Metal (mps).
func backendDevice(size, sizeVRAM int64) string {
	gpu := "cuda"
	if runtime.GOOS == "darwin" {
		gpu = "mps"
	}
	switch {
	case size <= 0:
		return "unknown"
	case sizeVRAM <= 0:
		return "cpu"
	case sizeVRAM >= size:
		return gpu
	default:
		return "mixed"
	}
}

// ShowInfo is the subset of /api/show the pre-flight check needs.
type ShowInfo struct {
	ParameterSize  string
	Quantization   string
	Family         string
	ContextLength  int
	EmbeddingDim   int
	BlockCount     int
	HeadCount      int
	HeadCountKV    int
	FileSizeBytes  int64 // from /api/tags if present
	AvailableLocal bool
}

// Show fetches model metadata via /api/show (and size via /api/tags). It
// returns AvailableLocal=false (no error) when the model is not pulled.
func (a *OllamaAdapter) Show(ctx context.Context, model string) (ShowInfo, error) {
	var info ShowInfo
	body, _ := json.Marshal(map[string]string{"model": model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.BaseURL+"/api/show", bytesReader(body))
	if err != nil {
		return info, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.Client.Do(req)
	if err != nil {
		return info, fmt.Errorf("%w: ollama: %v", core.ErrRuntimeUnavailable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return info, nil
	}
	if resp.StatusCode != http.StatusOK {
		return info, fmt.Errorf("ollama /api/show: status %d", resp.StatusCode)
	}
	var raw struct {
		Details struct {
			ParameterSize     string `json:"parameter_size"`
			QuantizationLevel string `json:"quantization_level"`
			Family            string `json:"family"`
		} `json:"details"`
		ModelInfo map[string]any `json:"model_info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return info, err
	}
	info.AvailableLocal = true
	info.ParameterSize = raw.Details.ParameterSize
	info.Quantization = raw.Details.QuantizationLevel
	info.Family = raw.Details.Family
	// model_info keys are prefixed by architecture, e.g. "llama.block_count".
	for k, v := range raw.ModelInfo {
		n, ok := v.(float64)
		if !ok {
			continue
		}
		switch {
		case hasSuffix(k, ".context_length"):
			info.ContextLength = int(n)
		case hasSuffix(k, ".embedding_length"):
			info.EmbeddingDim = int(n)
		case hasSuffix(k, ".block_count"):
			info.BlockCount = int(n)
		case hasSuffix(k, ".attention.head_count"):
			info.HeadCount = int(n)
		case hasSuffix(k, ".attention.head_count_kv"):
			info.HeadCountKV = int(n)
		}
	}
	// File size from /api/tags.
	var tags struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := a.getJSON(ctx, "/api/tags", &tags); err == nil {
		for _, m := range tags.Models {
			if m.Name == model || m.Name == model+":latest" {
				info.FileSizeBytes = m.Size
			}
		}
	}
	return info, nil
}

func (a *OllamaAdapter) getJSON(ctx context.Context, path string, dst any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.BaseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := a.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(dst)
}
