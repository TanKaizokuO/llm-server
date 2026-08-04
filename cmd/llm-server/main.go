// Command llm-server runs the Supervisor daemon or utility subcommands.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/httpserve"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

// defaultAddr is Ollama's conventional address. Existing clients point here
// without being reconfigured, which is the whole reason the Ollama surface
// exists. It is loopback-only: this is a single-operator tool with no
// authentication, so exposing it beyond the machine has to be deliberate.
const defaultAddr = "127.0.0.1:11434"

func main() {
	if err := runCLI(context.Background(), os.Args, os.Stdout); err != nil {
		slog.Error("command failed", "err", err)
		os.Exit(1)
	}
}

func runCLI(parentCtx context.Context, args []string, stdout io.Writer) error {
	if len(args) > 1 && args[1] == "inspect" {
		return runInspect(args[2:], stdout)
	}

	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	addr := fs.String("addr", defaultAddr, "address to serve the Ollama, OpenAI, and native surfaces on")
	if err := fs.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	return runServer(parentCtx, *addr, fs.Args()...)
}

func runInspect(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "output metadata as JSON")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	if fs.NArg() < 1 {
		return errors.New("usage: llm-server inspect [-json] <file.gguf>")
	}

	filePath := fs.Arg(0)
	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("opening GGUF file: %w", err)
	}
	defer f.Close()

	hdr, err := gguf.ReadHeader(bufio.NewReader(f))
	if err != nil {
		return fmt.Errorf("reading GGUF metadata: %w", err)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(hdr.Metadata)
	}

	fmt.Fprintf(stdout, "Architecture:   %s\n", hdr.Metadata.Architecture)
	fmt.Fprintf(stdout, "Context Length: %d\n", hdr.Metadata.ContextLength)
	fmt.Fprintf(stdout, "Quantization:   %s\n", hdr.Metadata.Quantization)
	return nil
}

func runServer(parentCtx context.Context, addr string, dirs ...string) error {
	if len(dirs) == 0 {
		dirs = []string{"."}
	}
	h := host.New()
	sup, err := supervisor.New(h, dirs...)
	if err != nil {
		return fmt.Errorf("initialising supervisor: %w", err)
	}
	defer func() { _ = sup.Close() }()
	ctx, stop := signal.NotifyContext(parentCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv, err := httpserve.Listen(addr, sup.Handler())
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
