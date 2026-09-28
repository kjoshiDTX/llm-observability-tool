// Command llm-obs is the CLI for a running llm-obs collector.
//
//	llm-obs status
//	llm-obs traces [--name N] [--window 1h]
//	llm-obs trace <id>
//	llm-obs diagnose <id>            (or: diagnose --slowest [--name N])
//	llm-obs compare --before v1 --after v2 [--name N] [--tag-key version]
//	llm-obs check --model llama3.2:3b --context 8192
//	llm-obs events [--since 1h]
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/core/diagnose"
	"github.com/krishjoshi/llm-obs/collector/internal/core/preflight"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

var (
	baseURL  string
	jsonOut  bool
	stdout   io.Writer = os.Stdout
	exitFunc           = os.Exit
)

func main() {
	if len(os.Args) < 2 {
		usage()
		exitFunc(2)
		return
	}
	cmd, args := os.Args[1], os.Args[2:]
	if err := run(cmd, args); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		exitFunc(1)
	}
}

func usage() {
	fmt.Fprintln(stdout, `llm-obs - local LLM observability CLI

Commands:
  status                              runtimes, loaded models, current snapshot
  traces   [--name N] [--window 1h]   latency percentiles grouped by trace name
  trace    <id>                       one trace with its system context
  diagnose <id> | --slowest [--name N] [--window 1h]
                                      plain-language explanation of why a call was slow
  compare  --before v1 --after v2 [--name N] [--tag-key version] [--window 24h]
                                      A/B regression report between two tagged sets
  check    --model M [--context T]    pre-flight: will this model fit?
  events   [--since 1h]               runtime/model state changes

Global flags (before the command's own flags): --url http://localhost:8765  --json`)
}

func newFlags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.StringVar(&baseURL, "url", envOr("LLM_OBS_URL", "http://localhost:8765"), "collector URL")
	fs.BoolVar(&jsonOut, "json", false, "print raw JSON")
	return fs
}

func run(cmd string, args []string) error {
	switch cmd {
	case "status":
		fs := newFlags(cmd)
		if err := fs.Parse(args); err != nil {
			return err
		}
		return cmdStatus()
	case "traces":
		fs := newFlags(cmd)
		name := fs.String("name", "", "trace name filter")
		window := fs.String("window", "1h", "look-back window")
		model := fs.String("model", "", "model filter")
		if err := fs.Parse(args); err != nil {
			return err
		}
		return cmdTraces(*name, *model, *window)
	case "trace":
		fs := newFlags(cmd)
		if err := fs.Parse(args); err != nil {
			return err
		}
		if fs.NArg() != 1 {
			return fmt.Errorf("usage: llm-obs trace <id>")
		}
		return cmdTrace(fs.Arg(0))
	case "diagnose":
		fs := newFlags(cmd)
		slowest := fs.Bool("slowest", false, "diagnose the slowest trace in the window")
		name := fs.String("name", "", "trace name filter (with --slowest)")
		window := fs.String("window", "1h", "look-back window (with --slowest)")
		if err := fs.Parse(args); err != nil {
			return err
		}
		id := fs.Arg(0)
		if *slowest {
			var err error
			id, err = slowestTraceID(*name, *window)
			if err != nil {
				return err
			}
		}
		if id == "" {
			return fmt.Errorf("usage: llm-obs diagnose <id> | --slowest")
		}
		return cmdDiagnose(id)
	case "compare":
		fs := newFlags(cmd)
		before := fs.String("before", "", "selector for the baseline set (key=value or value)")
		after := fs.String("after", "", "selector for the candidate set")
		name := fs.String("name", "", "restrict to one trace name")
		key := fs.String("tag-key", "version", "tag key used when selectors are bare values")
		window := fs.String("window", "", "look-back window (default: all)")
		if err := fs.Parse(args); err != nil {
			return err
		}
		if *before == "" || *after == "" {
			return fmt.Errorf("--before and --after are required")
		}
		return cmdCompare(*before, *after, *name, *key, *window)
	case "check":
		fs := newFlags(cmd)
		model := fs.String("model", "", "model name (e.g. llama3.2:3b)")
		ctx := fs.Int("context", 0, "requested context length in tokens")
		if err := fs.Parse(args); err != nil {
			return err
		}
		if *model == "" {
			return fmt.Errorf("--model is required")
		}
		return cmdCheck(*model, *ctx)
	case "events":
		fs := newFlags(cmd)
		since := fs.String("since", "1h", "look-back window")
		if err := fs.Parse(args); err != nil {
			return err
		}
		return cmdEvents(*since)
	case "help", "-h", "--help":
		usage()
		return nil
	}
	usage()
	return fmt.Errorf("unknown command %q", cmd)
}

// ---- HTTP helpers ----

func get(path string, q url.Values, dst any) error {
	u := strings.TrimRight(baseURL, "/") + path
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(u)
	if err != nil {
		return fmt.Errorf("cannot reach collector at %s (%v). Is `llm-obs-collector` running?", baseURL, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		var e struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(body, &e)
		if e.Error == "" {
			e.Error = string(body)
		}
		return fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(e.Error))
	}
	if jsonOut {
		var buf bytes.Buffer
		_ = json.Indent(&buf, body, "", "  ")
		fmt.Fprintln(stdout, buf.String())
		return errPrinted
	}
	return json.Unmarshal(body, dst)
}

// errPrinted signals that --json already wrote the output.
var errPrinted = fmt.Errorf("printed")

func handle(err error) error {
	if err == errPrinted {
		return nil
	}
	return err
}

// ---- commands ----

func cmdStatus() error {
	var h struct {
		Version  string               `json:"version"`
		Runtimes []core.RuntimeStatus `json:"runtimes"`
		DB       map[string]int64     `json:"db"`
		Inflight int                  `json:"inflight_traces"`
	}
	if err := get("/health", nil, &h); err != nil {
		return handle(err)
	}
	var snaps []core.RuntimeSnapshot
	if err := get("/snapshot", nil, &snaps); err != nil {
		return handle(err)
	}
	fmt.Fprintf(stdout, "collector %s at %s   db: %d snapshots, %d traces, %d events   in-flight traces: %d\n\n",
		h.Version, baseURL, h.DB["snapshots"], h.DB["traces"], h.DB["events"], h.Inflight)
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "RUNTIME\tSTATUS\tMODELS\tLAST POLL")
	for _, r := range h.Runtimes {
		st := "down"
		if r.Up {
			st = "up"
		}
		models := strings.Join(r.Models, ", ")
		if models == "" {
			models = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", r.Runtime, st, models, ago(r.LastPoll))
	}
	tw.Flush()
	if len(snaps) > 0 {
		fmt.Fprintln(stdout)
		tw = tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "MODEL\tRUNTIME\tDEVICE\tTOK/S\tMODEL MEM\tHOST MEM\tKV%\tCTX MAX\tRUNNING")
		for _, s := range snaps {
			tps := "-"
			if s.TokensPerSec > 0 {
				tps = fmt.Sprintf("%.1f (%s)", s.TokensPerSec, s.TPSSource)
			}
			mem := "-"
			if s.ModelMemMB > 0 {
				mem = fmt.Sprintf("%s (%s on %s)", mb(s.ModelMemMB), mb(s.ModelGPUMemMB), s.BackendDevice)
			}
			host := fmt.Sprintf("%s / %s", mb(s.MemUsedMB), mb(s.MemTotalMB))
			if s.UnifiedMemory {
				host += " unified"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%.0f\t%d\t%d\n", s.Model, s.Runtime, s.BackendDevice, tps, mem, host, s.KVCacheUsedPct, s.ContextMaxTok, s.RequestsRunning)
		}
		tw.Flush()
	}
	return nil
}

func cmdTraces(name, model, window string) error {
	q := url.Values{"window": {window}}
	if name != "" {
		q.Set("name", name)
	}
	if model != "" {
		q.Set("model", model)
	}
	var out struct {
		Window string        `json:"window"`
		Groups []trace.Stats `json:"groups"`
	}
	if err := get("/traces", q, &out); err != nil {
		return handle(err)
	}
	if len(out.Groups) == 0 {
		fmt.Fprintf(stdout, "no traces in the last %s\n", out.Window)
		return nil
	}
	fmt.Fprintf(stdout, "traces in the last %s\n\n", out.Window)
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tCOUNT\tERR\tP50\tP95\tP99\tMEAN\tTOK/S P50\tTTFT P50\tMODELS")
	for _, g := range out.Groups {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%.1f\t%s\t%s\n", g.Name, g.Count, g.Errors, ms(g.P50MS), ms(g.P95MS), ms(g.P99MS), ms(g.MeanMS), g.TPSP50, ms(g.TTFTP50MS), strings.Join(g.Models, ","))
	}
	tw.Flush()
	return nil
}

func cmdTrace(id string) error {
	var t trace.Trace
	if err := get("/traces/"+url.PathEscape(id), nil, &t); err != nil {
		return handle(err)
	}
	printTrace(t)
	return nil
}

func printTrace(t trace.Trace) {
	fmt.Fprintf(stdout, "trace %s  %s  model=%s runtime=%s\n", t.ID, t.Name, t.Model, t.Runtime)
	fmt.Fprintf(stdout, "  started %s  duration %s", t.StartTime.Format(time.RFC3339), ms(t.DurationMS))
	if t.TTFTMS > 0 {
		fmt.Fprintf(stdout, "  ttft %s", ms(t.TTFTMS))
	}
	fmt.Fprintf(stdout, "  tokens %d prompt + %d completion  %.1f tok/s\n", t.PromptTokens, t.CompletionTokens, t.TokensPerSec)
	if len(t.Tags) > 0 {
		fmt.Fprintf(stdout, "  tags: %s\n", tags(t.Tags))
	}
	if t.Error != "" {
		fmt.Fprintf(stdout, "  error: %s\n", t.Error)
	}
	c := t.Context
	fmt.Fprintf(stdout, "  system during call: %d snapshots  device=%s  host mem %.0f%%  kv-cache max %.0f%%  context %d/%d (%.0f%%)  concurrent traces %d\n",
		c.SnapshotCount, orDash(c.BackendDevice), c.MemPressure*100, c.KVCacheMaxPct, c.ContextUsedTok, c.ContextMaxTok, c.ContextPressure*100, c.ConcurrentTraces)
}

func slowestTraceID(name, window string) (string, error) {
	q := url.Values{"window": {window}, "limit": {"1000"}}
	if name != "" {
		q.Set("name", name)
	}
	saved := jsonOut
	jsonOut = false
	var ts []trace.Trace
	err := get("/traces/list", q, &ts)
	jsonOut = saved
	if err != nil {
		return "", err
	}
	if len(ts) == 0 {
		return "", fmt.Errorf("no traces in the last %s", window)
	}
	sort.Slice(ts, func(i, j int) bool { return ts[i].DurationMS > ts[j].DurationMS })
	return ts[0].ID, nil
}

func cmdDiagnose(id string) error {
	var d diagnose.Diagnosis
	if err := get("/traces/"+url.PathEscape(id)+"/diagnosis", nil, &d); err != nil {
		return handle(err)
	}
	fmt.Fprintf(stdout, "%s  %s  model=%s runtime=%s  %s at %.1f tok/s\n\n", strings.ToUpper(d.Severity), d.Name, d.Model, d.Runtime, ms(d.DurationMS), d.TPS)
	fmt.Fprintln(stdout, wrap(d.Summary, 100))
	if len(d.Causes) > 1 {
		fmt.Fprintln(stdout, "\nContributing causes:")
		for _, c := range d.Causes[1:] {
			fmt.Fprintf(stdout, "  - [%s, %.0f%%] %s\n", c.Kind, c.Confidence*100, wrap(c.Explanation, 96))
		}
	}
	for _, c := range d.Causes {
		if c.Suggestion != "" {
			fmt.Fprintf(stdout, "\nSuggestion (%s): %s\n", c.Kind, c.Suggestion)
			break
		}
	}
	ev := d.Evidence
	fmt.Fprintln(stdout, "\nEvidence:")
	if b := ev.Baseline; b != nil {
		fmt.Fprintf(stdout, "  baseline: %d prior calls, p50 %s, p95 %s, median %.1f tok/s, typical device %s -> this call %.2fx duration, %.2fx tok/s\n",
			b.Count, ms(b.P50MS), ms(b.P95MS), b.TPSMedian, orDash(b.Device), b.DurationRatio, b.TPSRatio)
	} else {
		fmt.Fprintln(stdout, "  baseline: none yet (fewer than 5 prior calls with this name+model)")
	}
	c := ev.Context
	fmt.Fprintf(stdout, "  context:  device=%s  host mem %.0f%% (%s/%s)  kv-cache max %.0f%%  context %d/%d tokens (%.0f%%)  concurrent traces %d  runtime requests running %d\n",
		orDash(c.BackendDevice), c.MemPressure*100, mb(c.MemUsedMB), mb(c.MemTotalMB), c.KVCacheMaxPct, c.ContextUsedTok, c.ContextMaxTok, c.ContextPressure*100, c.ConcurrentTraces, c.RequestsRunning)
	if len(ev.ConcurrentTraces) > 0 {
		fmt.Fprintf(stdout, "  concurrent traces:")
		for i, r := range ev.ConcurrentTraces {
			if i >= 5 {
				fmt.Fprintf(stdout, " ... (+%d)", len(ev.ConcurrentTraces)-5)
				break
			}
			fmt.Fprintf(stdout, " %s(%s %s)", r.ID[:min(8, len(r.ID))], r.Name, ms(r.DurationMS))
		}
		fmt.Fprintln(stdout)
	}
	if len(ev.Events) > 0 {
		fmt.Fprintln(stdout, "  events in window:")
		for _, e := range ev.Events {
			fmt.Fprintf(stdout, "    %s  %s %s %s\n", e.Timestamp.Format("15:04:05"), e.Kind, e.Model, e.Detail)
		}
	}
	if len(ev.Snapshots) > 0 {
		fmt.Fprintf(stdout, "  snapshots (%d):\n", len(ev.Snapshots))
		tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "    TIME\tDEVICE\tTOK/S\tHOST MEM\tMODEL GPU/TOTAL\tKV%\tRUNNING")
		for i, s := range ev.Snapshots {
			if i >= 12 {
				fmt.Fprintf(tw, "    ... (+%d more)\n", len(ev.Snapshots)-12)
				break
			}
			fmt.Fprintf(tw, "    %s\t%s\t%.1f\t%.0f%%\t%s/%s\t%.0f\t%d\n", s.Timestamp.Format("15:04:05.000"), s.BackendDevice, s.TokensPerSec, s.MemPressure()*100, mb(s.ModelGPUMemMB), mb(s.ModelMemMB), s.KVCacheUsedPct, s.RequestsRunning)
		}
		tw.Flush()
	}
	return nil
}

func cmdCompare(before, after, name, key, window string) error {
	q := url.Values{"before": {before}, "after": {after}, "tag_key": {key}}
	if name != "" {
		q.Set("name", name)
	}
	if window != "" {
		q.Set("window", window)
	}
	var c diagnose.Comparison
	if err := get("/compare", q, &c); err != nil {
		return handle(err)
	}
	title := "all traces"
	if c.Name != "" {
		title = c.Name
	}
	fmt.Fprintf(stdout, "A/B comparison: %s   before=%s (%d traces)   after=%s (%d traces)\n\n", title, tags(c.Before.Selector), c.Before.Count, tags(c.After.Selector), c.After.Count)
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "METRIC\tBEFORE\tAFTER\tDELTA\t")
	for _, d := range c.Deltas {
		if d.Before == 0 && d.After == 0 {
			continue
		}
		mark := ""
		if d.Pct != 0 {
			if d.Better {
				mark = "better"
			} else {
				mark = "worse"
			}
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%+.1f%%\t%s\n", d.Metric, fmtMetric(d.Metric, d.Before), fmtMetric(d.Metric, d.After), d.Pct, mark)
	}
	tw.Flush()
	if len(c.Patterns) > 0 {
		fmt.Fprintln(stdout, "\nDIAGNOSIS PATTERNS")
		tw = tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "PATTERN\tBEFORE\tAFTER\tNOTE")
		for _, p := range c.Patterns {
			fmt.Fprintf(tw, "%s\t%d (%.0f%%)\t%d (%.0f%%)\t%s\n", p.Kind, p.BeforeCount, p.BeforePct, p.AfterCount, p.AfterPct, p.Note)
		}
		tw.Flush()
	}
	fmt.Fprintf(stdout, "\nVerdict: %s\n", wrap(c.Verdict, 100))
	return nil
}

func cmdCheck(model string, ctx int) error {
	q := url.Values{"model": {model}, "context": {fmt.Sprint(ctx)}}
	var r preflight.Result
	if err := get("/check", q, &r); err != nil {
		return handle(err)
	}
	fmt.Fprintln(stdout, wrap(r.Verdict, 100))
	fmt.Fprintln(stdout)
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintf(tw, "model\t%s (%s, %.2fB params, %s)\n", r.Model, r.ParameterSize, r.Params, r.Quantization)
	fmt.Fprintf(tw, "requested context\t%d tokens", r.ContextTok)
	if r.ModelMaxContext > 0 {
		fmt.Fprintf(tw, " (model max %d)", r.ModelMaxContext)
	}
	fmt.Fprintln(tw)
	fmt.Fprintf(tw, "weights\t%s\n", mb(r.WeightsMB))
	fmt.Fprintf(tw, "kv cache\t%s (%d bytes/token x %d)\n", mb(r.KVCacheMB), r.KVBytesPerTok, r.ContextTok)
	fmt.Fprintf(tw, "overhead\t%s\n", mb(r.OverheadMB))
	fmt.Fprintf(tw, "required\t%s\n", mb(r.RequiredMB))
	fmt.Fprintf(tw, "host memory\t%s available of %s", mb(r.HostAvailableMB), mb(r.HostTotalMB))
	if r.UnifiedMemory {
		fmt.Fprintf(tw, " (unified; accelerator budget %s)", mb(r.GPUBudgetMB))
	}
	fmt.Fprintln(tw)
	if r.OtherResidentMB > 0 {
		fmt.Fprintf(tw, "other resident models\t%s\n", mb(r.OtherResidentMB))
	}
	fmt.Fprintf(tw, "usable for this model\t%s\n", mb(r.AvailableMB))
	fmt.Fprintf(tw, "margin\t%s\n", mb(r.MarginMB))
	if r.SuggestedContextTok > 0 {
		fmt.Fprintf(tw, "suggested context\t%d tokens\n", r.SuggestedContextTok)
	}
	fmt.Fprintf(tw, "estimate source\t%s\n", r.Source)
	tw.Flush()
	if len(r.Assumptions) > 0 {
		fmt.Fprintln(stdout, "\nAssumptions (heuristics, not measurements):")
		for _, a := range r.Assumptions {
			fmt.Fprintf(stdout, "  - %s\n", a)
		}
	}
	return nil
}

func cmdEvents(since string) error {
	var evs []core.Event
	if err := get("/events", url.Values{"since": {since}}, &evs); err != nil {
		return handle(err)
	}
	if len(evs) == 0 {
		fmt.Fprintf(stdout, "no events in the last %s\n", since)
		return nil
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "TIME\tKIND\tRUNTIME\tMODEL\tDETAIL")
	for _, e := range evs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", e.Timestamp.Format("2006-01-02 15:04:05"), e.Kind, e.Runtime, orDash(e.Model), e.Detail)
	}
	tw.Flush()
	return nil
}

// ---- formatting ----

func ms(v float64) string {
	if v == 0 {
		return "-"
	}
	if v >= 1000 {
		return fmt.Sprintf("%.2fs", v/1000)
	}
	return fmt.Sprintf("%.0fms", v)
}

func mb(v int64) string {
	if v >= 1024 {
		return fmt.Sprintf("%.1fGB", float64(v)/1024)
	}
	return fmt.Sprintf("%dMB", v)
}

func fmtMetric(name string, v float64) string {
	switch {
	case strings.HasSuffix(name, "_ms"):
		return ms(v)
	case name == "error_rate":
		return fmt.Sprintf("%.1f%%", v*100)
	default:
		return fmt.Sprintf("%.1f", v)
	}
}

func ago(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t).Round(time.Second)
	return d.String() + " ago"
}

func tags(m map[string]string) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+m[k])
	}
	return strings.Join(parts, ",")
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func wrap(s string, width int) string {
	words := strings.Fields(s)
	var b strings.Builder
	line := 0
	for _, w := range words {
		if line+len(w)+1 > width && line > 0 {
			b.WriteString("\n")
			line = 0
		} else if line > 0 {
			b.WriteString(" ")
			line++
		}
		b.WriteString(w)
		line += len(w)
	}
	return b.String()
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
