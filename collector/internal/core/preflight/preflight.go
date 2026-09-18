// Package preflight estimates whether a model will fit in memory at a given
// context length BEFORE it is loaded, so users stop discovering the answer by
// OOMing.
//
// Everything here is a documented heuristic, not a prediction:
//
//	weights   = file size if the model is pulled locally, else
//	            params x bytes-per-param(quantization)
//	kv cache  = 2 (K and V) x layers x kv_heads x head_dim x 2 bytes (f16) x context
//	            (architecture from /api/show when available, else a table
//	            keyed on parameter count)
//	overhead  = compute buffers + runtime: 8% of weights + 256 MB, min 384 MB
//	budget    = min(host memory currently available,
//	                GPU working-set limit (75% of unified memory on Apple
//	                Silicon) minus other resident models)
package preflight

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/hostmem"
)

// ModelInfo is runtime-provided metadata about a model (Ollama /api/show).
type ModelInfo struct {
	AvailableLocal bool
	ParameterSize  string // "8.0B"
	Quantization   string // "Q4_K_M"
	ContextLength  int
	EmbeddingDim   int
	BlockCount     int
	HeadCount      int
	HeadCountKV    int
	FileSizeBytes  int64
}

// ModelInfoFunc fetches ModelInfo; nil means "no runtime available to ask".
type ModelInfoFunc func(ctx context.Context, model string) (ModelInfo, error)

// Result is the pre-flight verdict with every number behind it.
type Result struct {
	Model         string  `json:"model"`
	ParameterSize string  `json:"parameter_size"`
	Params        float64 `json:"params_billions"`
	Quantization  string  `json:"quantization"`
	ContextTok    int     `json:"context_tok"`
	Source        string  `json:"source"` // where the estimate came from

	WeightsMB     int64 `json:"weights_mb"`
	KVCacheMB     int64 `json:"kv_cache_mb"`
	KVBytesPerTok int64 `json:"kv_bytes_per_token"`
	OverheadMB    int64 `json:"overhead_mb"`
	RequiredMB    int64 `json:"required_mb"`

	HostTotalMB     int64 `json:"host_total_mb"`
	HostAvailableMB int64 `json:"host_available_mb"`
	GPUBudgetMB     int64 `json:"gpu_budget_mb"`
	OtherResidentMB int64 `json:"other_resident_mb"`
	AvailableMB     int64 `json:"available_mb"` // the number we compare against
	UnifiedMemory   bool  `json:"unified_memory"`
	AlreadyLoaded   bool  `json:"already_loaded"`
	ModelMaxContext int   `json:"model_max_context,omitempty"`

	Fits                bool     `json:"fits"`
	MarginMB            int64    `json:"margin_mb"`
	SuggestedContextTok int      `json:"suggested_context_tok"`
	Verdict             string   `json:"verdict"`
	Assumptions         []string `json:"assumptions"`
}

// Preflight performs checks.
type Preflight struct {
	col  *core.Collector
	info ModelInfoFunc
	Host func() (hostmem.Stats, error)

	// GPUFraction is the share of unified memory the accelerator may use.
	// Metal's recommendedMaxWorkingSetSize is ~75% on Apple Silicon.
	GPUFraction float64
	// Safety is the fraction of the budget kept free.
	Safety float64
	// Now is overridable for tests.
	Now func() time.Time
}

// New creates a checker. col and info may be nil.
func New(col *core.Collector, info ModelInfoFunc) *Preflight {
	return &Preflight{col: col, info: info, Host: hostmem.Read, GPUFraction: 0.75, Safety: 0.05, Now: time.Now}
}

// Check implements api.Checker.
func (p *Preflight) Check(model string, contextTok int) (any, error) {
	return p.Estimate(context.Background(), model, contextTok), nil
}

// Estimate produces the Result.
func (p *Preflight) Estimate(ctx context.Context, model string, contextTok int) Result {
	r := Result{Model: model, ContextTok: contextTok}
	var info ModelInfo
	if p.info != nil {
		ictx, cancel := context.WithTimeout(ctx, 3*time.Second)
		i, err := p.info(ictx, model)
		cancel()
		if err == nil {
			info = i
		}
	}

	// ---- parameters and quantization ----
	r.Params, r.ParameterSize = paramsFrom(info.ParameterSize, model)
	r.Quantization = info.Quantization
	if r.Quantization == "" {
		r.Quantization = quantFromName(model)
	}
	if r.Quantization == "" {
		r.Quantization = "Q4_K_M"
		r.Assumptions = append(r.Assumptions, "quantization not specified; assuming Q4_K_M (Ollama default)")
	}
	r.ModelMaxContext = info.ContextLength
	if contextTok <= 0 {
		if info.ContextLength > 0 {
			contextTok = min(info.ContextLength, 4096)
		} else {
			contextTok = 4096
		}
		r.ContextTok = contextTok
		r.Assumptions = append(r.Assumptions, fmt.Sprintf("no --context given; using %d", contextTok))
	}
	if info.ContextLength > 0 && contextTok > info.ContextLength {
		r.Assumptions = append(r.Assumptions, fmt.Sprintf("requested context %d exceeds the model's trained maximum %d", contextTok, info.ContextLength))
	}

	// ---- weights ----
	switch {
	case info.FileSizeBytes > 0:
		r.WeightsMB = info.FileSizeBytes / (1024 * 1024)
		r.Source = "ollama /api/show + file size"
	case r.Params > 0:
		r.WeightsMB = int64(r.Params * 1e9 * bytesPerParam(r.Quantization) / (1024 * 1024))
		if info.AvailableLocal {
			r.Source = "ollama /api/show metadata"
		} else {
			r.Source = "name heuristic (model not pulled locally)"
			r.Assumptions = append(r.Assumptions, "parameter count parsed from the model name")
		}
	default:
		r.Source = "unknown"
		r.Verdict = fmt.Sprintf("Cannot estimate %q: no local metadata and no parameter count in the name (try e.g. llama3.2:3b).", model)
		return r
	}

	// ---- KV cache ----
	if info.BlockCount > 0 && info.HeadCountKV > 0 && info.EmbeddingDim > 0 && info.HeadCount > 0 {
		headDim := info.EmbeddingDim / info.HeadCount
		r.KVBytesPerTok = int64(2 * info.BlockCount * info.HeadCountKV * headDim * 2)
	} else {
		r.KVBytesPerTok = kvBytesPerTokenHeuristic(r.Params)
		r.Assumptions = append(r.Assumptions, fmt.Sprintf("architecture unknown; assuming %d KB/token of KV cache for a %.1fB model", r.KVBytesPerTok/1024, r.Params))
	}
	r.Assumptions = append(r.Assumptions, "KV cache assumed f16 (2 bytes/value); q8 KV cache would halve it")
	r.KVCacheMB = r.KVBytesPerTok * int64(contextTok) / (1024 * 1024)

	// ---- overhead ----
	r.OverheadMB = int64(float64(r.WeightsMB)*0.08) + 256
	if r.OverheadMB < 384 {
		r.OverheadMB = 384
	}
	r.RequiredMB = r.WeightsMB + r.KVCacheMB + r.OverheadMB

	// ---- available memory ----
	if p.Host != nil {
		if h, err := p.Host(); err == nil {
			r.HostTotalMB, r.HostAvailableMB, r.UnifiedMemory = h.TotalMB, h.AvailableMB, h.Unified
		}
	}
	if p.col != nil {
		for _, s := range p.col.Current() {
			if s.Model == model {
				r.AlreadyLoaded = true
				continue
			}
			r.OtherResidentMB += s.ModelGPUMemMB
			if s.ModelGPUMemMB == 0 {
				r.OtherResidentMB += s.ModelMemMB
			}
		}
	}
	if r.UnifiedMemory {
		r.GPUBudgetMB = int64(float64(r.HostTotalMB) * p.GPUFraction)
		r.Assumptions = append(r.Assumptions, fmt.Sprintf("unified memory: accelerator budget capped at %.0f%% of RAM (Metal working-set limit)", p.GPUFraction*100))
	} else {
		r.GPUBudgetMB = r.HostTotalMB
		r.Assumptions = append(r.Assumptions, "discrete-GPU VRAM is not probed; comparing against host memory")
	}
	r.AvailableMB = r.HostAvailableMB
	if b := r.GPUBudgetMB - r.OtherResidentMB; b < r.AvailableMB {
		r.AvailableMB = b
	}
	if r.AvailableMB < 0 {
		r.AvailableMB = 0
	}
	usable := int64(float64(r.AvailableMB) * (1 - p.Safety))
	r.MarginMB = usable - r.RequiredMB
	r.Fits = r.MarginMB >= 0

	// ---- suggestion ----
	if !r.Fits {
		room := usable - r.WeightsMB - r.OverheadMB
		if room > 0 && r.KVBytesPerTok > 0 {
			ctxFit := int(room * 1024 * 1024 / r.KVBytesPerTok)
			ctxFit = ctxFit / 512 * 512
			if ctxFit >= 512 {
				r.SuggestedContextTok = ctxFit
			}
		}
	}

	// ---- verdict ----
	gb := func(mb int64) string { return fmt.Sprintf("%.1f GB", float64(mb)/1024) }
	switch {
	case r.Fits && r.AlreadyLoaded:
		r.Verdict = fmt.Sprintf("YES - %s is already resident; %s at %d context needs ~%s and ~%s is usable.", model, r.Quantization, contextTok, gb(r.RequiredMB), gb(usable))
	case r.Fits:
		r.Verdict = fmt.Sprintf("YES - %s (%s, %.1fB params) at %d context needs ~%s (weights %s + KV %s + overhead %s); ~%s usable, %s to spare.",
			model, r.Quantization, r.Params, contextTok, gb(r.RequiredMB), gb(r.WeightsMB), gb(r.KVCacheMB), gb(r.OverheadMB), gb(usable), gb(r.MarginMB))
	case r.SuggestedContextTok > 0:
		r.Verdict = fmt.Sprintf("NO - %s at %d context needs ~%s but only ~%s is usable (short by %s). A context of %d tokens would fit.",
			model, contextTok, gb(r.RequiredMB), gb(usable), gb(-r.MarginMB), r.SuggestedContextTok)
	default:
		r.Verdict = fmt.Sprintf("NO - %s does not fit at any context: weights + overhead alone need ~%s but only ~%s is usable. Use a smaller model or lower quantization.",
			model, gb(r.WeightsMB+r.OverheadMB), gb(usable))
	}
	return r
}

var paramRe = regexp.MustCompile(`(?i)(?:^|[:\-_/])(\d+(?:\.\d+)?)\s*([bm])(?:[:\-_]|$)`)

// paramsFrom parses "8.0B" / "494.03M" or a model name like "llama3.2:3b".
func paramsFrom(paramSize, name string) (float64, string) {
	parse := func(s string) (float64, bool) {
		s = strings.TrimSpace(strings.ToUpper(s))
		if s == "" {
			return 0, false
		}
		mult := 1.0
		switch {
		case strings.HasSuffix(s, "B"):
			s = strings.TrimSuffix(s, "B")
		case strings.HasSuffix(s, "M"):
			s, mult = strings.TrimSuffix(s, "M"), 1e-3
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, false
		}
		return v * mult, true
	}
	if v, ok := parse(paramSize); ok {
		return v, paramSize
	}
	if m := paramRe.FindStringSubmatch(name); m != nil {
		if v, ok := parse(m[1] + m[2]); ok {
			return v, strings.ToUpper(m[1] + m[2])
		}
	}
	return 0, ""
}

var quantRe = regexp.MustCompile(`(?i)(q[2-8](?:_[k0-9](?:_[sml])?)?|fp16|f16|bf16|fp32|f32|int8|int4)`)

func quantFromName(name string) string {
	if m := quantRe.FindString(name); m != "" {
		return strings.ToUpper(m)
	}
	return ""
}

// bytesPerParam maps a quantization label to an effective bytes/param
// including the embedding/output layers that are often kept at higher
// precision. Values are calibrated against llama.cpp GGUF sizes.
func bytesPerParam(q string) float64 {
	q = strings.ToUpper(q)
	switch {
	case strings.HasPrefix(q, "Q2"):
		return 0.40
	case strings.HasPrefix(q, "Q3"):
		return 0.50
	case strings.HasPrefix(q, "Q4"), q == "INT4":
		return 0.60
	case strings.HasPrefix(q, "Q5"):
		return 0.72
	case strings.HasPrefix(q, "Q6"):
		return 0.85
	case strings.HasPrefix(q, "Q8"), q == "INT8":
		return 1.10
	case q == "F16", q == "FP16", q == "BF16":
		return 2.05
	case q == "F32", q == "FP32":
		return 4.05
	}
	return 0.60
}

// kvBytesPerTokenHeuristic approximates 2*layers*kv_heads*head_dim*2 for
// common architectures by parameter count (GQA assumed for >= 3B).
func kvBytesPerTokenHeuristic(paramsB float64) int64 {
	switch {
	case paramsB <= 0:
		return 64 * 1024
	case paramsB < 1:
		return 12 * 1024 // qwen2.5-0.5b: 24L x 2kv x 64d
	case paramsB < 2:
		return 32 * 1024 // llama3.2-1b: 16L x 8kv x 64d
	case paramsB < 5:
		return 112 * 1024 // llama3.2-3b: 28L x 8kv x 128d
	case paramsB < 10:
		return 128 * 1024 // llama3.1-8b: 32L x 8kv x 128d
	case paramsB < 20:
		return 192 * 1024
	case paramsB < 40:
		return 256 * 1024
	case paramsB < 80:
		return 320 * 1024 // llama3.1-70b: 80L x 8kv x 128d
	default:
		return int64(math.Round(paramsB/70*320)) * 1024
	}
}
