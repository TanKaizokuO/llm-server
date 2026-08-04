// Command llm-server runs the Supervisor daemon.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/TanKaizokuO/llm-server/internal/httpserve"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

// defaultAddr is Ollama's conventional address. Existing clients point here
// without being reconfigured, which is the whole reason the Ollama surface
// exists. It is loopback-only: this is a single-operator tool with no
// authentication, so exposing it beyond the machine has to be deliberate.
const defaultAddr = "127.0.0.1:11434"

func main() {
	addr := flag.String("addr", defaultAddr, "address to serve the Ollama, OpenAI, and native surfaces on")
	flag.Parse()

	if err := run(context.Background(), *addr); err != nil {
		slog.Error("supervisor failed", "err", err)
		os.Exit(1)
	}
}

func run(parentCtx context.Context, addr string) error {
	ctx, stop := signal.NotifyContext(parentCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv, err := httpserve.Listen(addr, supervisor.New().Handler())
	if err != nil {
		return fmt.Errorf("initialising server: %w", err)
	}

	slog.Info("supervisor listening", "addr", srv.Addr())
	err = srv.Serve(ctx)

	// Unregister signal notify immediately after Serve finishes so a second
	// signal won't be swallowed if callers use run in a process wrapper.
	stop()

	if err != nil {
		return fmt.Errorf("serving: %w", err)
	}
	slog.Info("supervisor stopped gracefully")
	return nil
}
