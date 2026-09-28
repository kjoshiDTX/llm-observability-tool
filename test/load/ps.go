package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// psSample reads instantaneous CPU% and RSS (KB) for pid via `ps`, which
// exists on both macOS and Linux with compatible -o column names. %cpu from
// ps is a point-in-time/recent-average figure (not a precise interval
// measurement), which is adequate for a "does this stay cheap" check.
func psSample(pid int) (cpuPct float64, rssKB int64, err error) {
	out, err := exec.Command("ps", "-o", "%cpu=,rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0, 0, fmt.Errorf("ps: %w", err)
	}
	fields := strings.Fields(strings.TrimSpace(string(out)))
	if len(fields) < 2 {
		return 0, 0, fmt.Errorf("ps: unexpected output %q", out)
	}
	cpuPct, err = strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, 0, err
	}
	rssKB, err = strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0, 0, err
	}
	return cpuPct, rssKB, nil
}
