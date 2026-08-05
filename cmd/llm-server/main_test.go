package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
)

func createTestModelFile(t *testing.T, dir, filename string) string {
	t.Helper()
	path := filepath.Join(dir, filename)
	data := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  "llama",
		ContextLength: 4096,
		Quantization:  "Q4_K_M",
	})
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("writing test GGUF: %v", err)
	}
	return path
}

func TestRunGracefulShutdownOnSignal(t *testing.T) {
	tmpDir := t.TempDir()
	createTestModelFile(t, tmpDir, "test-model.gguf")

	errCh := make(chan error, 1)
	go func() {
		errCh <- runServer(context.Background(), "127.0.0.1:0", filepath.Join(tmpDir, "tuning.json"), tmpDir)
	}()

	time.Sleep(50 * time.Millisecond)

	// Send real SIGINT signal to current process to test signal.NotifyContext
	if err := syscall.Kill(os.Getpid(), syscall.SIGINT); err != nil {
		t.Fatalf("sending SIGINT: %v", err)
	}

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("runServer returned error on SIGINT: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runServer did not exit within 5s of SIGINT")
	}
}

func TestRunFailsOnInvalidAddress(t *testing.T) {
	tmpDir := t.TempDir()
	createTestModelFile(t, tmpDir, "test-model.gguf")

	err := runServer(context.Background(), "invalid-address-format:99999999", filepath.Join(tmpDir, "tuning.json"), tmpDir)
	if err == nil {
		t.Fatal("expected error on invalid address, got nil")
	}
}

func TestRunServer_FailsWhenNoModelsFound(t *testing.T) {
	emptyDir := t.TempDir()
	err := runServer(context.Background(), "127.0.0.1:0", filepath.Join(emptyDir, "tuning.json"), emptyDir)
	if err == nil {
		t.Fatal("expected error when no models found, got nil")
	}
	if !strings.Contains(err.Error(), "no models found") {
		t.Errorf("err = %v, want error containing 'no models found'", err)
	}
}

func TestInspectSubcommand_ValidGGUF(t *testing.T) {
	tmpDir := t.TempDir()
	ggufPath := filepath.Join(tmpDir, "test.gguf")

	data := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  "llama",
		ContextLength: 4096,
		Quantization:  "Q4_K_M",
	})
	if err := os.WriteFile(ggufPath, data, 0600); err != nil {
		t.Fatalf("writing test gguf: %v", err)
	}

	var buf bytes.Buffer
	args := []string{"llm-server", "inspect", ggufPath}
	if err := runCLI(context.Background(), args, &buf); err != nil {
		t.Fatalf("unexpected error running inspect: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "Architecture:   llama") {
		t.Errorf("expected architecture output, got:\n%s", out)
	}
	if !strings.Contains(out, "Context Length: 4096") {
		t.Errorf("expected context length output, got:\n%s", out)
	}
	if !strings.Contains(out, "Quantization:   Q4_K_M") {
		t.Errorf("expected quantization output, got:\n%s", out)
	}
}

func TestInspectSubcommand_JSONOutput(t *testing.T) {
	tmpDir := t.TempDir()
	ggufPath := filepath.Join(tmpDir, "test.gguf")

	data := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  "qwen2",
		ContextLength: 32768,
		Quantization:  "Q8_0",
	})
	if err := os.WriteFile(ggufPath, data, 0600); err != nil {
		t.Fatalf("writing test gguf: %v", err)
	}

	var buf bytes.Buffer
	args := []string{"llm-server", "inspect", "-json", ggufPath}
	if err := runCLI(context.Background(), args, &buf); err != nil {
		t.Fatalf("unexpected error running inspect -json: %v", err)
	}

	var meta gguf.Metadata
	if err := json.Unmarshal(buf.Bytes(), &meta); err != nil {
		t.Fatalf("invalid json output (%v): %s", err, buf.String())
	}

	if meta.Architecture != "qwen2" {
		t.Errorf("expected architecture qwen2, got %q", meta.Architecture)
	}
	if meta.ContextLength != 32768 {
		t.Errorf("expected context length 32768, got %d", meta.ContextLength)
	}
	if meta.Quantization != "Q8_0" {
		t.Errorf("expected quantization Q8_0, got %q", meta.Quantization)
	}
}

func TestInspectSubcommand_MissingFile(t *testing.T) {
	var buf bytes.Buffer
	args := []string{"llm-server", "inspect", "nonexistent.gguf"}
	err := runCLI(context.Background(), args, &buf)
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestInspectSubcommand_MissingArgs(t *testing.T) {
	var buf bytes.Buffer
	args := []string{"llm-server", "inspect"}
	err := runCLI(context.Background(), args, &buf)
	if err == nil {
		t.Fatal("expected usage error for inspect with no arguments, got nil")
	}
}
