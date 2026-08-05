//go:build !linux

package host

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

func detectSystemMemory() (int64, error) {
	if runtime.GOOS == "darwin" {
		cmd := exec.Command("sysctl", "-n", "hw.memsize")
		out, err := cmd.Output()
		if err == nil {
			mem, parseErr := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
			if parseErr == nil && mem > 0 {
				return mem, nil
			}
		}
	}
	return 0, fmt.Errorf("unable to detect system memory")
}

func getProcessRAM(pid int) int64 {
	return 0
}

func isOOMSignal(waitErr error) bool {
	return false
}
