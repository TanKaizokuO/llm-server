package host

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"
)

// RealHost implements Host using actual system subprocesses and physical hardware.
type RealHost struct{}

// New constructs a RealHost process supervisor.
func New() *RealHost {
	return &RealHost{}
}

// NewRealHost is an explicit constructor for RealHost.
func NewRealHost() *RealHost {
	return &RealHost{}
}

// Fingerprint returns a stable identifier for the host hardware.
func (r *RealHost) Fingerprint() string {
	return "host-real-fingerprint"
}

// Launch starts a llama-server Instance using os/exec.
func (r *RealHost) Launch(ctx context.Context, argv []string) (Instance, error) {
	finalArgv, port, err := prepareArgv(argv)
	if err != nil {
		return nil, fmt.Errorf("preparing argv: %w", err)
	}

	cmd := exec.CommandContext(context.Background(), finalArgv[0], finalArgv[1:]...)
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

	go func() {
		err := cmd.Wait()
		inst.mu.Lock()
		inst.exitErr = err
		inst.mu.Unlock()
		close(done)
	}()

	return inst, nil
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
	cmd     *exec.Cmd
	port    int
	url     *url.URL
	done    chan struct{}
	once    sync.Once
	mu      sync.Mutex
	exitErr error
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
	return Allocation{
		VRAM: 0,
		RAM:  0,
	}
}

func (r *realInstance) Stop(ctx context.Context) error {
	r.once.Do(func() {
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
