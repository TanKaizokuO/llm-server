// Package httpserve owns the socket lifecycle for the daemon's HTTP surface:
// binding an address, serving a handler on it, and releasing it cleanly.
//
// It is deliberately separate from the supervisor package and contains no
// domain vocabulary. In this project a "server" otherwise means llama-server,
// which the glossary calls an Instance; keeping generic net/http plumbing out
// of the domain package prevents that collision and leaves each package with
// one reason to change.
package httpserve

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"
)

const (
	// shutdownGrace bounds how long a graceful shutdown waits for in-flight
	// requests before the remaining connections are closed underneath their
	// clients.
	shutdownGrace = 10 * time.Second

	// readHeaderTimeout stops a client that opens a connection and never
	// finishes sending headers from pinning a goroutine and a descriptor.
	readHeaderTimeout = 10 * time.Second

	// idleTimeout reclaims keep-alive connections a client has abandoned.
	idleTimeout = 120 * time.Second
)

// Server binds an address and serves a handler on it until its context is
// cancelled. Binding happens in Listen rather than in Serve so that a
// configured address that cannot be taken fails immediately and visibly, and
// so that a caller asking for port 0 can discover what it actually got.
type Server struct {
	listener net.Listener
	http     *http.Server
}

// Listen binds addr without serving on it yet.
func Listen(addr string, h http.Handler) (*Server, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("binding %s: %w", addr, err)
	}
	return &Server{
		listener: ln,
		http: &http.Server{
			Handler:           h,
			ReadHeaderTimeout: readHeaderTimeout,
			IdleTimeout:       idleTimeout,
			// WriteTimeout is deliberately left unset: this server streams
			// generation responses, which have no bounded duration.
		},
	}, nil
}

// Addr reports the address actually bound, which differs from the one
// requested whenever the caller asked the kernel to choose a port.
func (s *Server) Addr() string {
	return s.listener.Addr().String()
}

// Serve serves until ctx is cancelled, then stops accepting, lets in-flight
// requests finish within the grace period, and releases the address. It
// returns nil on a clean shutdown.
func (s *Server) Serve(ctx context.Context) error {
	// Cancelling on return retires the watcher below even when Serve fails
	// before the caller's context is ever cancelled.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	stopped := make(chan error, 1)
	go func() {
		<-ctx.Done()
		grace, cancelGrace := context.WithTimeout(context.WithoutCancel(ctx), shutdownGrace)
		defer cancelGrace()
		stopped <- s.http.Shutdown(grace)
	}()

	if err := s.http.Serve(s.listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serving on %s: %w", s.Addr(), err)
	}
	if err := <-stopped; err != nil {
		return fmt.Errorf("shutting down %s: %w", s.Addr(), err)
	}
	return nil
}
