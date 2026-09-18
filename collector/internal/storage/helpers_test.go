package storage

import "context"

func contextWithCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

// path is a test helper: recover the DSN we opened with.
func (s *Store) path() string { return s.dsn }
