// Package storage persists snapshots, events and traces in a local SQLite
// database (pure-Go driver, no CGO, no external service).
package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"github.com/krishjoshi/llm-obs/collector/internal/core"
	"github.com/krishjoshi/llm-obs/collector/internal/trace"
)

// Store wraps the SQLite database.
type Store struct {
	db  *sql.DB
	dsn string

	snapCh chan core.RuntimeSnapshot
	wg     sync.WaitGroup
	once   sync.Once
}

const schema = `
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  runtime TEXT NOT NULL, model TEXT NOT NULL,
  tokens_per_sec REAL, tps_source TEXT,
  mem_used_mb INTEGER, mem_total_mb INTEGER, unified INTEGER,
  model_mem_mb INTEGER, model_gpu_mem_mb INTEGER,
  kv_cache_pct REAL, ctx_used INTEGER, ctx_max INTEGER,
  req_running INTEGER, req_waiting INTEGER,
  device TEXT, param_size TEXT, quant TEXT
);
CREATE INDEX IF NOT EXISTS snapshots_key_ts ON snapshots(runtime, model, ts);
CREATE INDEX IF NOT EXISTS snapshots_ts ON snapshots(ts);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL, model TEXT NOT NULL, runtime TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '{}',
  start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL,
  duration_ms REAL NOT NULL, ttft_ms REAL,
  prompt_tokens INTEGER, completion_tokens INTEGER, tokens_per_sec REAL,
  error TEXT, context TEXT NOT NULL DEFAULT '{}',
  created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_name_start ON traces(name, start_ms);
CREATE INDEX IF NOT EXISTS traces_model_start ON traces(model, start_ms);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL, kind TEXT NOT NULL, runtime TEXT, model TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
`

// Open opens (creating if needed) the database at path. Use ":memory:" for
// tests.
func Open(path string) (*Store, error) {
	dsn := path
	if path == ":memory:" {
		dsn = "file::memory:?cache=shared"
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	// SQLite has a single writer; one connection avoids SQLITE_BUSY churn and
	// keeps in-memory databases coherent.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;"); err != nil {
		db.Close()
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return &Store{db: db, dsn: dsn, snapCh: make(chan core.RuntimeSnapshot, 4096)}, nil
}

// Close flushes buffered snapshots and closes the database.
func (s *Store) Close() error {
	s.once.Do(func() { close(s.snapCh) })
	s.wg.Wait()
	return s.db.Close()
}

// DB exposes the raw handle (tests, ad-hoc queries).
func (s *Store) DB() *sql.DB { return s.db }

// SnapshotHook returns a collector OnSnapshot function that buffers writes
// through a background batcher, so persistence never slows polling.
func (s *Store) SnapshotHook() func(core.RuntimeSnapshot) {
	return func(snap core.RuntimeSnapshot) {
		select {
		case s.snapCh <- snap:
		default: // buffer full: drop rather than block the poller
		}
	}
}

// StartWriter runs the batch writer until Close.
func (s *Store) StartWriter(ctx context.Context) {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		batch := make([]core.RuntimeSnapshot, 0, 256)
		flush := func() {
			if len(batch) == 0 {
				return
			}
			_ = s.InsertSnapshots(batch)
			batch = batch[:0]
		}
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			select {
			case snap, ok := <-s.snapCh:
				if !ok {
					flush()
					return
				}
				batch = append(batch, snap)
				if len(batch) >= 256 {
					flush()
				}
			case <-t.C:
				flush()
			case <-ctx.Done():
				// Drain what is already queued, then stop.
				for {
					select {
					case snap, ok := <-s.snapCh:
						if !ok {
							flush()
							return
						}
						batch = append(batch, snap)
					default:
						flush()
						return
					}
				}
			}
		}
	}()
}

// InsertSnapshots writes a batch in one transaction.
func (s *Store) InsertSnapshots(snaps []core.RuntimeSnapshot) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO snapshots (ts, runtime, model, tokens_per_sec, tps_source, mem_used_mb, mem_total_mb, unified,
		model_mem_mb, model_gpu_mem_mb, kv_cache_pct, ctx_used, ctx_max, req_running, req_waiting, device, param_size, quant)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, x := range snaps {
		if _, err := stmt.Exec(x.Timestamp.UnixMilli(), x.Runtime, x.Model, x.TokensPerSec, x.TPSSource, x.MemUsedMB, x.MemTotalMB, boolInt(x.UnifiedMemory),
			x.ModelMemMB, x.ModelGPUMemMB, x.KVCacheUsedPct, x.ContextUsedTok, x.ContextMaxTok, x.RequestsRunning, x.RequestsWaiting,
			x.BackendDevice, x.ParameterSize, x.Quantization); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// InsertSnapshot writes one snapshot synchronously.
func (s *Store) InsertSnapshot(x core.RuntimeSnapshot) error {
	return s.InsertSnapshots([]core.RuntimeSnapshot{x})
}

// SnapshotsInRange returns stored snapshots for a key in chronological order.
func (s *Store) SnapshotsInRange(runtime, model string, from, to time.Time) ([]core.RuntimeSnapshot, error) {
	rows, err := s.db.Query(`SELECT ts, runtime, model, tokens_per_sec, tps_source, mem_used_mb, mem_total_mb, unified,
		model_mem_mb, model_gpu_mem_mb, kv_cache_pct, ctx_used, ctx_max, req_running, req_waiting, device, param_size, quant
		FROM snapshots WHERE runtime=? AND model=? AND ts BETWEEN ? AND ? ORDER BY ts`,
		runtime, model, from.UnixMilli(), to.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []core.RuntimeSnapshot
	for rows.Next() {
		var x core.RuntimeSnapshot
		var ts int64
		var unified int
		if err := rows.Scan(&ts, &x.Runtime, &x.Model, &x.TokensPerSec, &x.TPSSource, &x.MemUsedMB, &x.MemTotalMB, &unified,
			&x.ModelMemMB, &x.ModelGPUMemMB, &x.KVCacheUsedPct, &x.ContextUsedTok, &x.ContextMaxTok, &x.RequestsRunning, &x.RequestsWaiting,
			&x.BackendDevice, &x.ParameterSize, &x.Quantization); err != nil {
			return nil, err
		}
		x.Timestamp = time.UnixMilli(ts)
		x.UnifiedMemory = unified == 1
		out = append(out, x)
	}
	return out, rows.Err()
}

// InsertEvent persists a collector event.
func (s *Store) InsertEvent(e core.Event) error {
	_, err := s.db.Exec(`INSERT INTO events (ts, kind, runtime, model, detail) VALUES (?,?,?,?,?)`,
		e.Timestamp.UnixMilli(), string(e.Kind), e.Runtime, e.Model, e.Detail)
	return err
}

// EventsInRange returns events in a window, oldest first.
func (s *Store) EventsInRange(from, to time.Time, runtime string) ([]core.Event, error) {
	q := `SELECT ts, kind, runtime, model, detail FROM events WHERE ts BETWEEN ? AND ?`
	args := []any{from.UnixMilli(), to.UnixMilli()}
	if runtime != "" {
		q += ` AND runtime=?`
		args = append(args, runtime)
	}
	rows, err := s.db.Query(q+` ORDER BY ts`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []core.Event
	for rows.Next() {
		var e core.Event
		var ts int64
		var kind string
		if err := rows.Scan(&ts, &kind, &e.Runtime, &e.Model, &e.Detail); err != nil {
			return nil, err
		}
		e.Kind, e.Timestamp = core.EventKind(kind), time.UnixMilli(ts)
		out = append(out, e)
	}
	return out, rows.Err()
}

// InsertTrace stores (or replaces) a trace.
func (s *Store) InsertTrace(t trace.Trace) error {
	tags, _ := json.Marshal(nonNil(t.Tags))
	ctxJSON, _ := json.Marshal(t.Context)
	_, err := s.db.Exec(`INSERT OR REPLACE INTO traces (id, name, model, runtime, tags, start_ms, end_ms, duration_ms, ttft_ms,
		prompt_tokens, completion_tokens, tokens_per_sec, error, context, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		t.ID, t.Name, t.Model, t.Runtime, string(tags), t.StartTime.UnixMilli(), t.EndTime.UnixMilli(), t.DurationMS, t.TTFTMS,
		t.PromptTokens, t.CompletionTokens, t.TokensPerSec, t.Error, string(ctxJSON), t.CreatedAt.UnixMilli())
	return err
}

const traceCols = `id, name, model, runtime, tags, start_ms, end_ms, duration_ms, ttft_ms, prompt_tokens, completion_tokens, tokens_per_sec, error, context, created_ms`

func scanTrace(sc interface{ Scan(...any) error }) (trace.Trace, error) {
	var t trace.Trace
	var tags, ctxJSON string
	var start, end, created int64
	var ttft sql.NullFloat64
	var errStr sql.NullString
	if err := sc.Scan(&t.ID, &t.Name, &t.Model, &t.Runtime, &tags, &start, &end, &t.DurationMS, &ttft,
		&t.PromptTokens, &t.CompletionTokens, &t.TokensPerSec, &errStr, &ctxJSON, &created); err != nil {
		return t, err
	}
	t.StartTime, t.EndTime, t.CreatedAt = time.UnixMilli(start), time.UnixMilli(end), time.UnixMilli(created)
	t.TTFTMS, t.Error = ttft.Float64, errStr.String
	_ = json.Unmarshal([]byte(tags), &t.Tags)
	_ = json.Unmarshal([]byte(ctxJSON), &t.Context)
	return t, nil
}

// GetTrace fetches one trace by id.
func (s *Store) GetTrace(id string) (trace.Trace, bool, error) {
	row := s.db.QueryRow(`SELECT `+traceCols+` FROM traces WHERE id=?`, id)
	t, err := scanTrace(row)
	if err == sql.ErrNoRows {
		return t, false, nil
	}
	return t, err == nil, err
}

// ListTraces returns traces matching q, newest first.
func (s *Store) ListTraces(q trace.Query) ([]trace.Trace, error) {
	var where []string
	var args []any
	if q.Name != "" {
		where = append(where, "name=?")
		args = append(args, q.Name)
	}
	if q.Model != "" {
		where = append(where, "model=?")
		args = append(args, q.Model)
	}
	if !q.Since.IsZero() {
		where = append(where, "start_ms>=?")
		args = append(args, q.Since.UnixMilli())
	}
	if !q.Until.IsZero() {
		where = append(where, "start_ms<=?")
		args = append(args, q.Until.UnixMilli())
	}
	for k, v := range q.Tags {
		where = append(where, "json_extract(tags, ?)=?")
		args = append(args, "$."+k, v)
	}
	sqlq := `SELECT ` + traceCols + ` FROM traces`
	if len(where) > 0 {
		sqlq += ` WHERE ` + strings.Join(where, " AND ")
	}
	sqlq += ` ORDER BY start_ms DESC`
	if q.Limit > 0 {
		sqlq += fmt.Sprintf(` LIMIT %d`, q.Limit)
	}
	rows, err := s.db.Query(sqlq, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []trace.Trace
	for rows.Next() {
		t, err := scanTrace(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// OverlappingTraces returns stored traces on the same model whose window
// intersects [start, end], excluding excludeID.
func (s *Store) OverlappingTraces(model string, start, end time.Time, excludeID string) ([]trace.Trace, error) {
	rows, err := s.db.Query(`SELECT `+traceCols+` FROM traces WHERE model=? AND id<>? AND start_ms<=? AND end_ms>=? ORDER BY start_ms`,
		model, excludeID, end.UnixMilli(), start.UnixMilli())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []trace.Trace
	for rows.Next() {
		t, err := scanTrace(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// Baseline returns up to limit completed, non-error traces with the same
// name and model that started before `before`, newest first.
func (s *Store) Baseline(name, model string, before time.Time, limit int) ([]trace.Trace, error) {
	rows, err := s.db.Query(`SELECT `+traceCols+` FROM traces WHERE name=? AND model=? AND start_ms<? AND (error IS NULL OR error='')
		ORDER BY start_ms DESC LIMIT ?`, name, model, before.UnixMilli(), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []trace.Trace
	for rows.Next() {
		t, err := scanTrace(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// Prune deletes snapshots and events older than the retention window.
// Traces are kept (they are small and are the product).
func (s *Store) Prune(retention time.Duration) error {
	cutoff := time.Now().Add(-retention).UnixMilli()
	if _, err := s.db.Exec(`DELETE FROM snapshots WHERE ts<?`, cutoff); err != nil {
		return err
	}
	_, err := s.db.Exec(`DELETE FROM events WHERE ts<?`, cutoff)
	return err
}

// Counts returns row counts for a quick health view.
func (s *Store) Counts() (snapshots, traces, events int64) {
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM snapshots`).Scan(&snapshots)
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM traces`).Scan(&traces)
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM events`).Scan(&events)
	return
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func nonNil(m map[string]string) map[string]string {
	if m == nil {
		return map[string]string{}
	}
	return m
}
