//go:build linux

package host

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

func detectSystemMemory() (int64, error) {
	f, err := os.Open("/proc/meminfo")
	if err == nil {
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					kb, parseErr := strconv.ParseInt(fields[1], 10, 64)
					if parseErr == nil && kb > 0 {
						return kb * 1024, nil
					}
				}
			}
		}
	}
	var info syscall.Sysinfo_t
	if err := syscall.Sysinfo(&info); err == nil {
		return int64(info.Totalram) * int64(info.Unit), nil
	}
	return 0, fmt.Errorf("unable to detect system memory")
}

func getProcessRAM(pid int) int64 {
	if pid <= 0 {
		return 0
	}
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/statm", pid))
	if err == nil {
		fields := strings.Fields(string(data))
		if len(fields) >= 2 {
			pages, parseErr := strconv.ParseInt(fields[1], 10, 64)
			if parseErr == nil && pages > 0 {
				return pages * int64(os.Getpagesize())
			}
		}
	}
	return 0
}

func isOOMSignal(waitErr error) bool {
	var exitErr *exec.ExitError
	if errors.As(waitErr, &exitErr) {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
			if status.Signaled() && status.Signal() == syscall.SIGKILL {
				return true
			}
		}
	}
	return false
}
