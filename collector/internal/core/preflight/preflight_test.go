package preflight

import (
	"context"
	"strings"
	"testing"

	"github.com/krishjoshi/llm-obs/collector/internal/hostmem"
)

func fixedHost(total, avail int64, unified bool) func() (hostmem.Stats, error) {
	return func() (hostmem.Stats, error) {
		return hostmem.Stats{TotalMB: total, AvailableMB: avail, UsedMB: total - avail, Unified: unified}, nil
	}
}

func qwenInfo(ctx context.Context, model string) (ModelInfo, error) {
	return ModelInfo{AvailableLocal: true, ParameterSize: "494.03M", Quantization: "Q4_K_M", ContextLength: 32768,
		EmbeddingDim: 896, BlockCount: 24, HeadCount: 14, HeadCountKV: 2, FileSizeBytes: 397821319}, nil
}

func TestFitsSmallModelFromLocalMetadata(t *testing.T) {
	p := New(nil, qwenInfo)
	p.Host = fixedHost(16384, 6000, true)
	r := p.Estimate(context.Background(), "qwen2.5:0.5b", 8192)
	if !r.Fits {
		t.Fatalf("expected fit: %+v", r)
	}
	if r.WeightsMB != 379 {
		t.Errorf("weights from file size = %d", r.WeightsMB)
	}
	// 2 * 24 layers * 2 kv heads * 64 dim * 2 bytes = 12288 B/token
	if r.KVBytesPerTok != 12288 {
		t.Errorf("kv bytes/token = %d", r.KVBytesPerTok)
	}
	if r.KVCacheMB != 96 {
		t.Errorf("kv cache = %d MB", r.KVCacheMB)
	}
	if !strings.HasPrefix(r.Verdict, "YES") || r.Source == "" {
		t.Errorf("verdict: %s", r.Verdict)
	}
}

func TestDoesNotFitSuggestsSmallerContext(t *testing.T) {
	// 8B model from name only, on a 16 GB machine with ~6 GB available.
	p := New(nil, nil)
	p.Host = fixedHost(16384, 6000, true)
	r := p.Estimate(context.Background(), "llama3.1:8b", 32768)
	if r.Fits {
		t.Fatalf("expected not to fit: %+v", r)
	}
	if r.Params != 8 || r.Quantization != "Q4_K_M" {
		t.Errorf("name parsing: %+v", r)
	}
	if r.SuggestedContextTok <= 0 || r.SuggestedContextTok >= 32768 || r.SuggestedContextTok%512 != 0 {
		t.Errorf("suggested context = %d", r.SuggestedContextTok)
	}
	if !strings.HasPrefix(r.Verdict, "NO") || !strings.Contains(r.Verdict, "would fit") {
		t.Errorf("verdict: %s", r.Verdict)
	}
	// The suggested context must itself fit.
	r2 := p.Estimate(context.Background(), "llama3.1:8b", r.SuggestedContextTok)
	if !r2.Fits {
		t.Errorf("suggested context %d does not fit: %+v", r.SuggestedContextTok, r2)
	}
}

func TestDoesNotFitAtAll(t *testing.T) {
	p := New(nil, nil)
	p.Host = fixedHost(16384, 4000, true)
	r := p.Estimate(context.Background(), "llama3.1:70b-instruct-q4_K_M", 2048)
	if r.Fits || r.SuggestedContextTok != 0 {
		t.Fatalf("70B on 16GB should never fit: %+v", r)
	}
	if !strings.Contains(r.Verdict, "any context") {
		t.Errorf("verdict: %s", r.Verdict)
	}
	if r.Quantization != "Q4_K_M" {
		t.Errorf("quant from name: %q", r.Quantization)
	}
}

func TestGPUBudgetCapsUnifiedMemory(t *testing.T) {
	p := New(nil, nil)
	// Plenty "available" but the 75% accelerator cap should bind.
	p.Host = fixedHost(16384, 16000, true)
	r := p.Estimate(context.Background(), "x:13b", 4096)
	if r.AvailableMB != 12288 {
		t.Errorf("available should be capped at 75%% of 16384 = 12288, got %d", r.AvailableMB)
	}
	p.Host = fixedHost(16384, 16000, false)
	r = p.Estimate(context.Background(), "x:13b", 4096)
	if r.AvailableMB != 16000 {
		t.Errorf("non-unified should not be capped: %d", r.AvailableMB)
	}
}

func TestUnparseableModel(t *testing.T) {
	p := New(nil, nil)
	p.Host = fixedHost(16384, 8000, true)
	r := p.Estimate(context.Background(), "mystery-model", 4096)
	if r.Fits || !strings.Contains(r.Verdict, "Cannot estimate") {
		t.Errorf("unexpected: %+v", r)
	}
}

func TestParamParsing(t *testing.T) {
	cases := map[string]float64{
		"llama3.2:3b": 3, "qwen2.5:0.5b": 0.5, "mistral:7b-instruct-q5_K_M": 7, "gemma2:27b": 27,
		"phi3:14b-medium-4k-instruct-q4_0": 14, "smollm:135m": 0.135, "llama3.2:latest": 0,
	}
	for name, want := range cases {
		if got, _ := paramsFrom("", name); got != want {
			t.Errorf("%s: got %v want %v", name, got, want)
		}
	}
	if v, _ := paramsFrom("8.0B", "whatever"); v != 8 {
		t.Error("explicit param size ignored")
	}
	if quantFromName("mistral:7b-instruct-q5_K_M") != "Q5_K_M" || quantFromName("llama3:8b-fp16") != "FP16" {
		t.Error("quant parsing")
	}
}

func TestDefaultContextWhenUnspecified(t *testing.T) {
	p := New(nil, qwenInfo)
	p.Host = fixedHost(16384, 8000, true)
	r := p.Estimate(context.Background(), "qwen2.5:0.5b", 0)
	if r.ContextTok != 4096 {
		t.Errorf("default context = %d", r.ContextTok)
	}
}
