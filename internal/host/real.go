package host

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// RealHost implements Host using actual system subprocesses and physical hardware.
type RealHost struct {
	acceleratorDetector func() ([]Accelerator, error)
	systemMemoryReader  func() (int64, error)
	llamaBuildIDReader  func() (string, error)
}

// Option configures a RealHost.
type Option func(*RealHost)

// WithAcceleratorDetector sets a custom accelerator detector.
func WithAcceleratorDetector(fn func() ([]Accelerator, error)) Option {
	return func(r *RealHost) {
		r.acceleratorDetector = fn
	}
}

// WithSystemMemoryReader sets a custom system memory reader.
func WithSystemMemoryReader(fn func() (int64, error)) Option {
	return func(r *RealHost) {
		r.systemMemoryReader = fn
	}
}

// WithLlamaBuildIDReader sets a custom llama-server build identifier reader.
func WithLlamaBuildIDReader(fn func() (string, error)) Option {
	return func(r *RealHost) {
		r.llamaBuildIDReader = fn
	}
}

// New constructs a RealHost process supervisor.
func New(opts ...Option) *RealHost {
	r := &RealHost{}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// NewRealHost is an explicit constructor for RealHost.
func NewRealHost(opts ...Option) *RealHost {
	return New(opts...)
}

// Accelerators returns the ordered list of visible compute devices on the host.
func (r *RealHost) Accelerators() []Accelerator {
	if r.acceleratorDetector != nil {
		accs, _ := r.acceleratorDetector()
		if accs != nil {
			return accs
		}
	}
	accs, _ := detectAccelerators()
	if accs == nil {
		return []Accelerator{}
	}
	return accs
}

func (r *RealHost) systemMemory() int64 {
	if r.systemMemoryReader != nil {
		mem, err := r.systemMemoryReader()
		if err == nil {
			return mem
		}
	}
	mem, _ := detectSystemMemory()
	return mem
}

func (r *RealHost) llamaBuildID() string {
	if r.llamaBuildIDReader != nil {
		id, err := r.llamaBuildIDReader()
		if err == nil && id != "" {
			return id
		}
	}
	return detectLlamaServerBuildID()
}

// Fingerprint returns a stable identifier for the host hardware.
func (r *RealHost) Fingerprint() string {
	accs := r.Accelerators()
	var accParts []string
	for _, a := range accs {
		accParts = append(accParts, fmt.Sprintf("id=%s,name=%s,mem=%d", a.ID, a.Name, a.TotalMemory))
	}
	accSummary := strings.Join(accParts, ";")
	if accSummary == "" {
		accSummary = "none"
	}
	sysMem := r.systemMemory()
	buildID := r.llamaBuildID()

	raw := fmt.Sprintf("accelerators:%s\nsystem_ram:%d\nllama_build:%s", accSummary, sysMem, buildID)
	sum := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", sum)
}

// Launch starts a llama-server Instance using os/exec.
func (r *RealHost) Launch(ctx context.Context, argv []string) (Instance, error) {
	finalArgv, port, err := prepareArgv(argv)
	if err != nil {
		return nil, fmt.Errorf("preparing argv: %w", err)
	}

	cmd := exec.CommandContext(ctx, finalArgv[0], finalArgv[1:]...)

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting subprocess %s: %w", finalArgv[0], err)
	}

	targetURL, err := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))
	if err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("parsing instance target url: %w", err)
	}

	done := make(chan struct{})
	inst := &realInstance{
		cmd:  cmd,
		port: port,
		url:  targetURL,
		done: done,
	}

	go inst.scanStderr(stderrPipe)

	go func() {
		waitErr := cmd.Wait()
		inst.mu.Lock()
		defer inst.mu.Unlock()
		if inst.stopped {
			inst.exitErr = nil
		} else if waitErr != nil {
			stderrStr := inst.stderrBuf.String()
			exitCode := -1
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			}
			isOOM := isOOMFailure(waitErr, stderrStr)
			inst.exitErr = &ProcessError{
				ExitCode: exitCode,
				OOM:      isOOM,
				Stderr:   stderrStr,
				Err:      waitErr,
			}
		} else {
			inst.exitErr = nil
		}
		close(done)
	}()

	return inst, nil
}

func detectAccelerators() ([]Accelerator, error) {
	if path, err := exec.LookPath("nvidia-smi"); err == nil && path != "" {
		cmd := exec.Command(path, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits")
		out, err := cmd.Output()
		if err == nil {
			var accs []Accelerator
			scanner := bufio.NewScanner(bytes.NewReader(out))
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line == "" {
					continue
				}
				parts := strings.Split(line, ",")
				if len(parts) >= 3 {
					idx := strings.TrimSpace(parts[0])
					name := strings.TrimSpace(parts[1])
					memMiBStr := strings.TrimSpace(parts[2])
					memMiB, parseErr := strconv.ParseInt(memMiBStr, 10, 64)
					if parseErr == nil && memMiB > 0 {
						accs = append(accs, Accelerator{
							ID:          fmt.Sprintf("cuda:%s", idx),
							Name:        name,
							TotalMemory: memMiB * 1024 * 1024,
						})
					}
				}
			}
			if len(accs) > 0 {
				return accs, nil
			}
		}
	}

	if runtime.GOOS == "linux" {
		matches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
		if err == nil && len(matches) > 0 {
			var accs []Accelerator
			for i, match := range matches {
				data, readErr := os.ReadFile(match)
				if readErr == nil {
					memBytes, parseErr := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
					if parseErr == nil && memBytes > 0 {
						accs = append(accs, Accelerator{
							ID:          fmt.Sprintf("gpu:%d", i),
							Name:        fmt.Sprintf("DRM GPU %d", i),
							TotalMemory: memBytes,
						})
					}
				}
			}
			if len(accs) > 0 {
				return accs, nil
			}
		}
	}

	return []Accelerator{}, nil
}

func detectLlamaServerBuildID() string {
	path, err := exec.LookPath("llama-server")
	if err != nil || path == "" {
		return "unknown"
	}

	cmd := exec.Command(path, "--version")
	out, err := cmd.CombinedOutput()
	if err == nil {
		s := strings.TrimSpace(string(out))
		if s != "" {
			lines := strings.Split(s, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line != "" {
					return line
				}
			}
		}
	}

	data, err := os.ReadFile(path)
	if err == nil && len(data) > 0 {
		sum := sha256.Sum256(data)
		return fmt.Sprintf("binary:%x", sum[:8])
	}

	return "unknown"
}

func isOOMFailure(waitErr error, stderr string) bool {
	if isOOMSignal(waitErr) {
		return true
	}

	lower := strings.ToLower(stderr)
	oomPatterns := []string{
		"out of memory",
		"cuda error: out of memory",
		"failed to allocate memory",
		"std::bad_alloc",
		"ggml_cuda_host_malloc: failed to allocate",
		"ggml_backend_buffer_init: failed to allocate",
		"not enough memory",
		"alloc_buffer: failed to allocate",
		"cannot allocate memory",
		"oom-killer",
	}

	for _, pattern := range oomPatterns {
		if strings.Contains(lower, pattern) {
			return true
		}
	}

	return false
}

func prepareArgv(argv []string) ([]string, int, error) {
	if len(argv) == 0 {
		return nil, 0, fmt.Errorf("empty argv")
	}

	finalArgv := make([]string, len(argv))
	copy(finalArgv, argv)

	portIdx := -1
	for i, arg := range finalArgv {
		if arg == "--port" || arg == "-port" {
			portIdx = i
			break
		}
	}

	var port int
	if portIdx != -1 && portIdx+1 < len(finalArgv) {
		p, err := strconv.Atoi(finalArgv[portIdx+1])
		if err == nil && p > 0 {
			port = p
		} else {
			freePort, err := getFreePort()
			if err != nil {
				return nil, 0, fmt.Errorf("allocating free port: %w", err)
			}
			port = freePort
			finalArgv[portIdx+1] = strconv.Itoa(port)
		}
	} else {
		freePort, err := getFreePort()
		if err != nil {
			return nil, 0, fmt.Errorf("allocating free port: %w", err)
		}
		port = freePort
		finalArgv = append(finalArgv, "--port", strconv.Itoa(port))
	}

	return finalArgv, port, nil
}

func getFreePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	return port, nil
}

type realInstance struct {
	cmd       *exec.Cmd
	port      int
	url       *url.URL
	done      chan struct{}
	once      sync.Once
	mu        sync.Mutex
	exitErr   error
	stopped   bool
	stderrBuf bytes.Buffer
	vramObs   int64
}

var vramRegexes = []*regexp.Regexp{
	regexp.MustCompile(`(?i)VRAM\s+total\s*=\s*([\d.]+)\s*(MiB|GiB|B)`),
	regexp.MustCompile(`(?i)VRAM\s+footprint\s*[:=]\s*([\d.]+)\s*(MiB|GiB|B)`),
	regexp.MustCompile(`(?i)VRAM\s+used\s*[:=]\s*([\d.]+)\s*(MiB|GiB|B)`),
	regexp.MustCompile(`(?i)offloaded\s+\d+/\d+\s+layers\s+to\s+GPU\s*,\s*([\d.]+)\s*(MiB|GiB|B)`),
	regexp.MustCompile(`(?i)VRAM\s+buffer\s+size\s*=\s*([\d.]+)\s*(MiB|GiB|B)`),
}

func (r *realInstance) scanStderr(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()

		r.mu.Lock()
		r.stderrBuf.WriteString(line)
		r.stderrBuf.WriteByte('\n')

		for _, re := range vramRegexes {
			matches := re.FindStringSubmatch(line)
			if len(matches) >= 3 {
				val, parseErr := strconv.ParseFloat(matches[1], 64)
				unit := strings.ToUpper(matches[2])
				if parseErr == nil && val > 0 {
					var b int64
					switch unit {
					case "GIB":
						b = int64(val * 1024 * 1024 * 1024)
					case "MIB":
						b = int64(val * 1024 * 1024)
					case "B":
						b = int64(val)
					}
					if b > r.vramObs {
						r.vramObs = b
					}
				}
			}
		}
		r.mu.Unlock()
	}
}

func (r *realInstance) WaitReady(ctx context.Context) error {
	client := &http.Client{
		Timeout: 500 * time.Millisecond,
	}
	healthURL := fmt.Sprintf("%s/health", r.url.String())

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-r.done:
			r.mu.Lock()
			defer r.mu.Unlock()
			if r.exitErr != nil {
				return fmt.Errorf("instance exited before ready: %w", r.exitErr)
			}
			return fmt.Errorf("instance exited before ready")
		case <-ticker.C:
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
			if err != nil {
				continue
			}
			resp, err := client.Do(req)
			if err == nil {
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
		}
	}
}

func (r *realInstance) URL() *url.URL {
	return r.url
}

func (r *realInstance) ObservedAllocation() Allocation {
	r.mu.Lock()
	vram := r.vramObs
	pid := 0
	if r.cmd != nil && r.cmd.Process != nil {
		pid = r.cmd.Process.Pid
	}
	r.mu.Unlock()

	ram := getProcessRAM(pid)
	return Allocation{
		VRAM: vram,
		RAM:  ram,
	}
}

func (r *realInstance) Stop(ctx context.Context) error {
	r.once.Do(func() {
		r.mu.Lock()
		r.stopped = true
		r.mu.Unlock()
		if r.cmd != nil && r.cmd.Process != nil {
			_ = r.cmd.Process.Signal(os.Interrupt)
		}
	})

	select {
	case <-r.done:
		return nil
	case <-ctx.Done():
		if r.cmd != nil && r.cmd.Process != nil {
			_ = r.cmd.Process.Kill()
		}
		return ctx.Err()
	}
}

func (r *realInstance) Done() <-chan struct{} {
	return r.done
}

func (r *realInstance) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.exitErr
}
