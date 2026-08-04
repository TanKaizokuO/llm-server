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
	"net/url"
)

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

	// Launch starts an Instance with the given argv.
	Launch(ctx context.Context, argv []string) (Instance, error)
}
