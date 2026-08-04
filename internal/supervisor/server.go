package supervisor

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"
)

// shutdownGrace bounds how long a graceful shutdown waits for in-flight
// requests before the remaining connections are closed underneath their
// clients.
const shutdownGrace = 10 * time.Second

// Server binds an address and serves a handler on it until its context is
// cancelled. Binding happens in Listen rather than in Serve so that a
// configured address that cannot be taken fails immediately and visibly,
// and so that a caller asking for port 0 can discover what it actually got.
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
		http:     &http.Server{Handler: h},
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
	stopped := make(chan error, 1)
	go func() {
		<-ctx.Done()
		grace, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownGrace)
		defer cancel()
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
