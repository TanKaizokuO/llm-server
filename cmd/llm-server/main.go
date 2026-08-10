// Command llm-server runs the Supervisor daemon or utility subcommands.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/TanKaizokuO/llm-server/internal/gguf"
	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/httpserve"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"
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

type serverOptions struct {
	Addr           string
	CachePath      string
	ConfigPath     string
	Budget         time.Duration
	IdleTTL        time.Duration
	RescanInterval time.Duration
	MaxInstances   int
	Slots          int
	Dirs           []string
}

func runCLI(parentCtx context.Context, args []string, stdout io.Writer) error {
	if len(args) > 1 && args[1] == "inspect" {
		return runInspect(args[2:], stdout)
	}

	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	addr := fs.String("addr", defaultAddr, "address to serve the Ollama, OpenAI, and native surfaces on")
	cachePath := fs.String("tuning-cache", "tuning.json", "path to the tuning cache JSON file")
	configPath := fs.String("config", "", "path to an optional JSON configuration file overriding per-model argv, TTL, slots, and tuned values")
	budget := fs.Duration("tuning-budget", 2*time.Minute, "maximum duration allowed for a single tuning run")
	idleTTL := fs.Duration("idle-ttl", 5*time.Minute, "default idle TTL for resident instances")
	rescanInterval := fs.Duration("rescan-interval", 30*time.Second, "interval for periodic background directory rescans (0 disables)")
	maxInstances := fs.Int("max-instances", 0, "maximum number of resident instances allowed (0 = uncapped)")
	slots := fs.Int("slots", 1, "number of concurrent slots per instance")
	if err := fs.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	opts := serverOptions{
		Addr:           *addr,
		CachePath:      *cachePath,
		ConfigPath:     *configPath,
		Budget:         *budget,
		IdleTTL:        *idleTTL,
		RescanInterval: *rescanInterval,
		MaxInstances:   *maxInstances,
		Slots:          *slots,
		Dirs:           fs.Args(),
	}
	return runServer(parentCtx, opts)
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

func resolveScanDirs(explicitDirs []string) []string {
	var scanDirs []string
	if len(explicitDirs) > 0 {
		scanDirs = append(scanDirs, explicitDirs...)
	} else {
		scanDirs = append(scanDirs, supervisor.ConventionalModelDirs()...)
	}
	return scanDirs
}

func runServer(parentCtx context.Context, opts serverOptions) error {
	scanDirs := resolveScanDirs(opts.Dirs)
	h := host.New()
	sup, err := supervisor.NewWithOpts(h, scanDirs,
		supervisor.WithCachePath(opts.CachePath),
		supervisor.WithConfigFile(opts.ConfigPath),
		supervisor.WithTuningBudget(opts.Budget),
		supervisor.WithDefaultTTL(opts.IdleTTL),
		supervisor.WithRescanInterval(opts.RescanInterval),
		supervisor.WithMaxInstances(opts.MaxInstances),
		supervisor.WithSlotsPerInstance(opts.Slots),
	)
	if err != nil {
		return fmt.Errorf("initialising supervisor: %w", err)
	}
	defer func() { _ = sup.Close() }()
	ctx, stop := signal.NotifyContext(parentCtx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv, err := httpserve.Listen(opts.Addr, sup.Handler())
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
