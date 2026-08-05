// Package host defines the Host boundary — the single injected boundary in the
// llm-server codebase.
//
// Host represents the physical machine: its hardware fingerprint, accelerator
// budget, and child subprocess supervision. Everything above Host is
// deterministic and directly testable; everything below it is a real
// subprocess and physical hardware.
package host

import (
	"context"
	"errors"
	"fmt"
	"net/url"
)

// Accelerator represents a physical compute accelerator device (e.g. GPU) on the host.
type Accelerator struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	TotalMemory int64  `json:"total_memory"`
}

// ProcessError represents a child process termination error with failure classification.
type ProcessError struct {
	ExitCode int
	OOM      bool
	Stderr   string
	Err      error
}

func (e *ProcessError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.OOM {
		if e.ExitCode != 0 {
			return fmt.Sprintf("instance terminated due to out-of-memory (exit code %d)", e.ExitCode)
		}
		return "instance terminated due to out-of-memory"
	}
	if e.ExitCode != 0 {
		return fmt.Sprintf("instance crashed (exit code %d)", e.ExitCode)
	}
	if e.Err != nil {
		return fmt.Sprintf("instance terminated: %v", e.Err)
	}
	return "instance terminated"
}

func (e *ProcessError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// IsOOM returns true if err represents an out-of-memory failure.
func IsOOM(err error) bool {
	if err == nil {
		return false
	}
	var pe *ProcessError
	if errors.As(err, &pe) {
		return pe.OOM
	}
	return false
}

// Allocation describes observed memory usage of a running Instance.
type Allocation struct {
	VRAM int64 `json:"vram"`
	RAM  int64 `json:"ram"`
}

// Instance represents a running llama-server child process.
type Instance interface {
	// WaitReady blocks until the Instance is ready to accept HTTP requests or ctx is cancelled.
	WaitReady(ctx context.Context) error

	// URL returns the HTTP base URL of the running Instance.
	URL() *url.URL

	// ObservedAllocation returns the observed memory usage of the Instance.
	ObservedAllocation() Allocation

	// Stop gracefully terminates the Instance.
	Stop(ctx context.Context) error

	// Done returns a channel that is closed when the Instance process terminates.
	Done() <-chan struct{}

	// Err returns the error if the Instance process terminated with an error, or nil on clean exit.
	Err() error
}

// Host represents the physical machine: hardware fingerprint and subprocess supervision.
// It is the single injected boundary in the llm-server codebase.
type Host interface {
	// Fingerprint returns a stable hardware identifier.
	Fingerprint() string

	// Accelerators returns the ordered list of visible compute devices on the host.
	Accelerators() []Accelerator

	// Launch starts an Instance with the given argv.
	Launch(ctx context.Context, argv []string) (Instance, error)
}
