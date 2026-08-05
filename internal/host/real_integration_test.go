//go:build integration

// Package host_test contains integration tests for the real Host implementation.
//
// Integration Test Exclusion Gap Notice:
// These tests execute against a physical GPU and a real llama-server binary.
// They are explicitly tagged with `//go:build integration` and excluded from default
// `go test ./...` test runs. This exclusion is a deliberate architectural gap:
// standard unit tests rely on FakeHost or helper subprocess mocks to maintain fast,
// deterministic CI suites, while physical hardware tests are run on demand or on GPU runners
// using `go test -tags integration ./...`.
package host_test

import (
	"context"
	"net/http"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
)

func TestRealHost_Integration_RealHardwareAndLlamaServer(t *testing.T) {
	llamaPath, err := exec.LookPath("llama-server")
	if err != nil || llamaPath == "" {
		t.Skip("llama-server binary not found on PATH; skipping real hardware integration test")
	}

	modelPath := os.Getenv("TEST_MODEL_PATH")
	if modelPath == "" {
		modelPath = os.Getenv("LLAMA_MODEL_PATH")
	}
	if modelPath == "" {
		t.Skip("TEST_MODEL_PATH environment variable not set; skipping real model launch integration test")
	}

	rh := host.New()
	fp := rh.Fingerprint()
	if fp == "" {
		t.Fatalf("RealHost.Fingerprint() returned empty string")
	}

	accs := rh.Accelerators()
	t.Logf("RealHost Fingerprint: %s", fp)
	t.Logf("RealHost Accelerators: %+v", accs)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	argv := []string{llamaPath, "-m", modelPath, "-c", "512", "-ngl", "99"}
	inst, err := rh.Launch(ctx, argv)
	if err != nil {
		t.Fatalf("RealHost.Launch failed: %v", err)
	}
	t.Cleanup(func() { _ = inst.Stop(context.Background()) })

	if err := inst.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady failed on real llama-server: %v", err)
	}

	resp, err := http.Get(inst.URL().String() + "/health")
	if err != nil {
		t.Fatalf("GET /health failed: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected health status code: %d", resp.StatusCode)
	}

	alloc := inst.ObservedAllocation()
	t.Logf("Observed allocation on real llama-server: VRAM=%d bytes, RAM=%d bytes", alloc.VRAM, alloc.RAM)

	if err := inst.Stop(ctx); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	select {
	case <-inst.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for instance Done channel after Stop")
	}

	if inst.Err() != nil {
		t.Errorf("unexpected instance error after clean Stop: %v", inst.Err())
	}
}
