// Package hostmem reports system-wide memory so snapshots can carry a
// memory-pressure signal regardless of what the runtime itself exposes.
package hostmem

import (
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v4/mem"
)

// Stats is a point-in-time host memory reading.
type Stats struct {
	TotalMB     int64
	UsedMB      int64
	AvailableMB int64
	Unified     bool
}

var (
	mu       sync.Mutex
	cached   Stats
	cachedAt time.Time
	// Probe is swappable in tests.
	Probe = probe
)

// Unified reports whether this host uses a unified CPU/GPU memory pool.
// Apple Silicon is the only such platform we target.
func Unified() bool { return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" }

func probe() (Stats, error) {
	vm, err := mem.VirtualMemory()
	if err != nil {
		return Stats{}, err
	}
	const mb = 1024 * 1024
	return Stats{
		TotalMB:     int64(vm.Total / mb),
		UsedMB:      int64((vm.Total - vm.Available) / mb),
		AvailableMB: int64(vm.Available / mb),
		Unified:     Unified(),
	}, nil
}

// Read returns host memory stats, cached for 500ms so several adapters
// polling at once do not each hit the kernel.
func Read() (Stats, error) {
	mu.Lock()
	defer mu.Unlock()
	if time.Since(cachedAt) < 500*time.Millisecond {
		return cached, nil
	}
	s, err := Probe()
	if err != nil {
		return cached, err
	}
	cached, cachedAt = s, time.Now()
	return s, nil
}
