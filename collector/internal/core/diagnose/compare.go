package diagnose

import (
	"fmt"
	"sort"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// SetSummary describes one side of an A/B comparison.
type SetSummary struct {
	Selector  map[string]string `json:"selector"`
	Count     int               `json:"count"`
	Errors    int               `json:"errors"`
	ErrorRate float64           `json:"error_rate"`
	P50MS     float64           `json:"p50_ms"`
	P95MS     float64           `json:"p95_ms"`
	P99MS     float64           `json:"p99_ms"`
	MeanMS    float64           `json:"mean_ms"`
	TPSP50    float64           `json:"tokens_per_sec_p50"`
	TPSMean   float64           `json:"tokens_per_sec_mean"`
	TTFTP50MS float64           `json:"ttft_p50_ms"`
	Models    []string          `json:"models"`
}

// Delta is the percentage change after vs before for one metric
// (positive = after is higher).
type Delta struct {
	Metric string  `json:"metric"`
	Before float64 `json:"before"`
	After  float64 `json:"after"`
	Pct    float64 `json:"pct"`
	Better bool    `json:"better"` // whether the direction is an improvement
}

// PatternDelta reports how often a diagnosis signal appeared in each set.
type PatternDelta struct {
	Kind        string  `json:"kind"`
	BeforeCount int     `json:"before_count"`
	AfterCount  int     `json:"after_count"`
	BeforePct   float64 `json:"before_pct"`
	AfterPct    float64 `json:"after_pct"`
	Note        string  `json:"note"`
}

// Comparison is the full A/B report.
type Comparison struct {
	Name     string         `json:"name,omitempty"`
	Before   SetSummary     `json:"before"`
	After    SetSummary     `json:"after"`
	Deltas   []Delta        `json:"deltas"`
	Patterns []PatternDelta `json:"patterns"`
	Verdict  string         `json:"verdict"`
}

// Lister is the store subset the comparer needs.
type Lister interface {
	ListTraces(q trace.Query) ([]trace.Trace, error)
}

// Comparer builds A/B reports from two tag-selected trace sets.
type Comparer struct {
	store Lister
	diag  *Diagnoser
	Limit int // max traces per side to diagnose
}

// NewComparer creates a comparer.
func NewComparer(store Lister, diag *Diagnoser) *Comparer {
	return &Comparer{store: store, diag: diag, Limit: 1000}
}

// Compare loads both sets and produces the report.
func (c *Comparer) Compare(before, after map[string]string, name string, since time.Time) (any, error) {
	bt, err := c.store.ListTraces(trace.Query{Name: name, Tags: before, Since: since, Limit: c.Limit})
	if err != nil {
		return nil, err
	}
	at, err := c.store.ListTraces(trace.Query{Name: name, Tags: after, Since: since, Limit: c.Limit})
	if err != nil {
		return nil, err
	}
	return c.CompareTraces(name, before, after, bt, at), nil
}

// CompareTraces is the pure computation over two already-loaded sets.
func (c *Comparer) CompareTraces(name string, bsel, asel map[string]string, bt, at []trace.Trace) Comparison {
	out := Comparison{Name: name, Before: summarize(bsel, bt), After: summarize(asel, at)}
	b, a := out.Before, out.After
	out.Deltas = []Delta{
		delta("latency_p50_ms", b.P50MS, a.P50MS, false),
		delta("latency_p95_ms", b.P95MS, a.P95MS, false),
		delta("latency_p99_ms", b.P99MS, a.P99MS, false),
		delta("tokens_per_sec_p50", b.TPSP50, a.TPSP50, true),
		delta("tokens_per_sec_mean", b.TPSMean, a.TPSMean, true),
		delta("ttft_p50_ms", b.TTFTP50MS, a.TTFTP50MS, false),
		delta("error_rate", b.ErrorRate, a.ErrorRate, false),
	}

	// Diagnosis-pattern frequencies.
	bp, ap := c.patterns(bt), c.patterns(at)
	kinds := map[string]bool{}
	for k := range bp {
		kinds[k] = true
	}
	for k := range ap {
		kinds[k] = true
	}
	var keys []string
	for k := range kinds {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		pd := PatternDelta{Kind: k, BeforeCount: bp[k], AfterCount: ap[k]}
		if len(bt) > 0 {
			pd.BeforePct = 100 * float64(bp[k]) / float64(len(bt))
		}
		if len(at) > 0 {
			pd.AfterPct = 100 * float64(ap[k]) / float64(len(at))
		}
		switch {
		case pd.BeforeCount == 0 && pd.AfterCount > 0:
			pd.Note = fmt.Sprintf("%s appeared in %.0f%% of 'after' runs vs 0%% in 'before' (new)", kindLabel(k), pd.AfterPct)
		case pd.AfterCount == 0 && pd.BeforeCount > 0:
			pd.Note = fmt.Sprintf("%s disappeared: %.0f%% of 'before' runs vs 0%% in 'after' (resolved)", kindLabel(k), pd.BeforePct)
		case pd.AfterPct > pd.BeforePct*1.5:
			pd.Note = fmt.Sprintf("%s became more frequent: %.0f%% -> %.0f%%", kindLabel(k), pd.BeforePct, pd.AfterPct)
		case pd.BeforePct > pd.AfterPct*1.5:
			pd.Note = fmt.Sprintf("%s became less frequent: %.0f%% -> %.0f%%", kindLabel(k), pd.BeforePct, pd.AfterPct)
		default:
			pd.Note = fmt.Sprintf("%s unchanged (%.0f%% vs %.0f%%)", kindLabel(k), pd.BeforePct, pd.AfterPct)
		}
		out.Patterns = append(out.Patterns, pd)
	}
	if out.Patterns == nil {
		out.Patterns = []PatternDelta{}
	}
	out.Verdict = verdict(out)
	return out
}

func (c *Comparer) patterns(ts []trace.Trace) map[string]int {
	counts := map[string]int{}
	if c.diag == nil {
		return counts
	}
	for _, t := range ts {
		d := c.diag.DiagnoseTrace(t)
		seen := map[string]bool{}
		for _, s := range d.Signals {
			if !seen[s.Kind] {
				seen[s.Kind] = true
				counts[s.Kind]++
			}
		}
		if d.Slow && len(d.Causes) > 0 && d.Causes[0].Kind == KindRegression {
			counts[KindRegression]++
		}
	}
	return counts
}

func summarize(sel map[string]string, ts []trace.Trace) SetSummary {
	st := trace.Summarize("", ts)
	s := SetSummary{Selector: sel, Count: st.Count, Errors: st.Errors, P50MS: st.P50MS, P95MS: st.P95MS, P99MS: st.P99MS,
		MeanMS: st.MeanMS, TPSP50: st.TPSP50, TPSMean: st.TPSMean, TTFTP50MS: st.TTFTP50MS, Models: st.Models}
	if st.Count > 0 {
		s.ErrorRate = float64(st.Errors) / float64(st.Count)
	}
	if s.Models == nil {
		s.Models = []string{}
	}
	return s
}

func delta(metric string, before, after float64, higherIsBetter bool) Delta {
	d := Delta{Metric: metric, Before: before, After: after}
	if before != 0 {
		d.Pct = (after - before) / before * 100
	} else if after != 0 {
		d.Pct = 100
	}
	if higherIsBetter {
		d.Better = after > before
	} else {
		d.Better = after < before
	}
	return d
}

func verdict(c Comparison) string {
	if c.Before.Count == 0 || c.After.Count == 0 {
		return fmt.Sprintf("Not enough data: before=%d traces, after=%d traces.", c.Before.Count, c.After.Count)
	}
	p95 := c.Deltas[1]
	tps := c.Deltas[3]
	var parts []string
	switch {
	case p95.Pct <= -10:
		parts = append(parts, fmt.Sprintf("'after' is %.0f%% faster at p95 (%.0f -> %.0f ms)", -p95.Pct, p95.Before, p95.After))
	case p95.Pct >= 10:
		parts = append(parts, fmt.Sprintf("'after' is %.0f%% slower at p95 (%.0f -> %.0f ms)", p95.Pct, p95.Before, p95.After))
	default:
		parts = append(parts, fmt.Sprintf("p95 latency roughly unchanged (%.0f -> %.0f ms, %+.0f%%)", p95.Before, p95.After, p95.Pct))
	}
	if tps.Before > 0 && tps.After > 0 {
		switch {
		case tps.Pct >= 10:
			parts = append(parts, fmt.Sprintf("throughput up %.0f%% (%.1f -> %.1f tok/s)", tps.Pct, tps.Before, tps.After))
		case tps.Pct <= -10:
			parts = append(parts, fmt.Sprintf("throughput down %.0f%% (%.1f -> %.1f tok/s)", -tps.Pct, tps.Before, tps.After))
		}
	}
	if c.After.ErrorRate > c.Before.ErrorRate {
		parts = append(parts, fmt.Sprintf("error rate rose %.1f%% -> %.1f%%", c.Before.ErrorRate*100, c.After.ErrorRate*100))
	}
	for _, p := range c.Patterns {
		if p.BeforeCount == 0 && p.AfterCount > 0 {
			parts = append(parts, p.Note)
		}
	}
	s := ""
	for i, p := range parts {
		if i > 0 {
			s += "; "
		}
		s += p
	}
	return s + "."
}
