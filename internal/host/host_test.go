package host_test

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
)

func TestFakeHost_BudgetAndFingerprint(t *testing.T) {
	fh := host.NewFakeHost()

	if fh.Fingerprint() != "fake-host-fingerprint" {
		t.Errorf("unexpected fingerprint: got %q", fh.Fingerprint())
	}

	fh.SetFingerprint("custom-fp")
	if fh.Fingerprint() != "custom-fp" {
		t.Errorf("unexpected custom fingerprint: got %q", fh.Fingerprint())
	}

	fh.SetBudget(16*1024*1024*1024, 32*1024*1024*1024, 250*1024*1024)
	if fh.VRAMBudget() != 16*1024*1024*1024 {
		t.Errorf("unexpected vram budget: %d", fh.VRAMBudget())
	}
	if fh.RAMBudget() != 32*1024*1024*1024 {
		t.Errorf("unexpected ram budget: %d", fh.RAMBudget())
	}
	if fh.LayerCost() != 250*1024*1024 {
		t.Errorf("unexpected layer cost: %d", fh.LayerCost())
	}
}

func TestFakeHost_LaunchAndDefaultHandler(t *testing.T) {
	ctx := context.Background()
	fh := host.NewFakeHost()

	argv := []string{"llama-server", "-m", "/models/test.gguf", "-c", "2048"}
	inst, err := fh.Launch(ctx, argv)
	if err != nil {
		t.Fatalf("Launch failed: %v", err)
	}
	t.Cleanup(func() { _ = inst.Stop(ctx) })

	if err := inst.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady failed: %v", err)
	}

	launches := fh.Launches()
	if len(launches) != 1 {
		t.Fatalf("expected 1 launch, got %d", len(launches))
	}
	if strings.Join(fh.LastLaunch(), " ") != strings.Join(argv, " ") {
		t.Errorf("unexpected last launch: got %v, want %v", fh.LastLaunch(), argv)
	}

	// Test health endpoint
	resp, err := http.Get(inst.URL().String() + "/health")
	if err != nil {
		t.Fatalf("GET /health failed: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"status":"ok"`) {
		t.Errorf("unexpected health response: status=%d body=%s", resp.StatusCode, string(body))
	}

	// Test streaming completions endpoint
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, inst.URL().String()+"/v1/chat/completions", strings.NewReader(`{"model":"test","stream":true}`))
	if err != nil {
		t.Fatalf("NewRequest failed: %v", err)
	}
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status code: %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			lines = append(lines, line)
		}
	}

	if len(lines) != 4 {
		t.Fatalf("expected 4 SSE lines, got %d: %v", len(lines), lines)
	}
	if !strings.Contains(lines[0], `"content":"Hello"`) {
		t.Errorf("unexpected line 0: %s", lines[0])
	}
	if !strings.Contains(lines[3], `[DONE]`) {
		t.Errorf("unexpected line 3: %s", lines[3])
	}

	alloc := inst.ObservedAllocation()
	if alloc.VRAM == 0 || alloc.RAM == 0 {
		t.Errorf("unexpected zero allocation: %+v", alloc)
	}

	if err := inst.Stop(ctx); err != nil {
		t.Errorf("Stop failed: %v", err)
	}

	select {
	case <-inst.Done():
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for inst.Done channel")
	}
}

func TestFakeHost_CustomOnLaunch(t *testing.T) {
	ctx := context.Background()
	fh := host.NewFakeHost()

	fh.SetOnLaunch(func(argv []string) (http.Handler, error) {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusTeapot)
			_, _ = w.Write([]byte("custom handler"))
		}), nil
	})

	inst, err := fh.Launch(ctx, []string{"custom-cmd"})
	if err != nil {
		t.Fatalf("Launch failed: %v", err)
	}
	t.Cleanup(func() { _ = inst.Stop(ctx) })

	resp, err := http.Get(inst.URL().String())
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusTeapot || string(body) != "custom handler" {
		t.Errorf("unexpected response: status=%d body=%s", resp.StatusCode, string(body))
	}
}

func TestRealHost_SuperviseHelperProcess(t *testing.T) {
	t.Setenv("GO_WANT_HELPER_PROCESS", "1")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	rh := host.New()
	if rh.Fingerprint() == "" {
		t.Errorf("empty fingerprint")
	}

	// Launch test binary using helper process pattern
	argv := []string{os.Args[0], "-test.run=TestHelperProcess", "--", "helper"}
	inst, err := rh.Launch(ctx, argv)
	if err != nil {
		t.Fatalf("Launch real host failed: %v", err)
	}
	t.Cleanup(func() { _ = inst.Stop(context.Background()) })

	if err := inst.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady real host failed: %v", err)
	}

	resp, err := http.Get(inst.URL().String() + "/health")
	if err != nil {
		t.Fatalf("GET /health failed: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"status":"ok"`) {
		t.Errorf("unexpected health response: status=%d body=%s", resp.StatusCode, string(body))
	}

	if err := inst.Stop(ctx); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	select {
	case <-inst.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for real instance Done channel")
	}
}

// TestHelperProcess acts as a mock llama-server process when invoked by TestRealHost unit tests.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	mode := os.Getenv("GO_WANT_HELPER_MODE")
	if mode == "oom" {
		fmt.Fprintf(os.Stderr, "llama_model_load: CUDA error: out of memory\n")
		os.Exit(1)
	}
	if mode == "crash" {
		fmt.Fprintf(os.Stderr, "llama_model_load: invalid model architecture 'foo'\n")
		os.Exit(1)
	}

	if mode == "alloc" {
		fmt.Fprintf(os.Stderr, "llama_model_load: VRAM total = 4096.00 MiB\n")
		fmt.Fprintf(os.Stderr, "llama_kv_cache_init: VRAM footprint = 512.00 MiB\n")
	}

	args := os.Args
	var port int
	for i, arg := range args {
		if (arg == "--port" || arg == "-port") && i+1 < len(args) {
			p, err := strconv.Atoi(args[i+1])
			if err == nil {
				port = p
			}
		}
	}

	if port == 0 {
		fmt.Fprintf(os.Stderr, "no port specified\n")
		os.Exit(1)
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		fmt.Fprintf(os.Stderr, "listening on %d failed: %v\n", port, err)
		os.Exit(1)
	}

	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/health" {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"status":"ok"}`))
				return
			}
			http.NotFound(w, r)
		}),
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		_ = srv.Serve(ln)
	}()

	<-ctx.Done()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer shutdownCancel()
	_ = srv.Shutdown(shutdownCtx)
	os.Exit(0)
}

func TestRealHost_FingerprintAndAccelerators(t *testing.T) {
	accDetector := func() ([]host.Accelerator, error) {
		return []host.Accelerator{
			{ID: "cuda:0", Name: "NVIDIA GeForce RTX 3090", TotalMemory: 24 * 1024 * 1024 * 1024},
		}, nil
	}
	sysMemReader := func() (int64, error) {
		return 64 * 1024 * 1024 * 1024, nil
	}
	buildIDReader := func() (string, error) {
		return "b4500", nil
	}

	rh := host.New(
		host.WithAcceleratorDetector(accDetector),
		host.WithSystemMemoryReader(sysMemReader),
		host.WithLlamaBuildIDReader(buildIDReader),
	)

	accs := rh.Accelerators()
	if len(accs) != 1 || accs[0].ID != "cuda:0" || accs[0].TotalMemory != 24*1024*1024*1024 {
		t.Errorf("unexpected accelerators: %+v", accs)
	}

	fp1 := rh.Fingerprint()
	if fp1 == "" {
		t.Fatalf("empty fingerprint")
	}

	fp2 := rh.Fingerprint()
	if fp1 != fp2 {
		t.Errorf("fingerprint not stable: %q vs %q", fp1, fp2)
	}

	rhDifferent := host.New(
		host.WithAcceleratorDetector(accDetector),
		host.WithSystemMemoryReader(sysMemReader),
		host.WithLlamaBuildIDReader(func() (string, error) { return "b4501", nil }),
	)
	if rhDifferent.Fingerprint() == fp1 {
		t.Errorf("fingerprint should change when build ID changes")
	}
}

func TestFakeHost_Accelerators(t *testing.T) {
	fh := host.NewFakeHost()
	accs := fh.Accelerators()
	if len(accs) != 1 || accs[0].ID != "gpu:0" {
		t.Errorf("unexpected default fake accelerators: %+v", accs)
	}

	custom := []host.Accelerator{
		{ID: "cuda:0", Name: "GPU 0", TotalMemory: 16 * 1024 * 1024 * 1024},
		{ID: "cuda:1", Name: "GPU 1", TotalMemory: 16 * 1024 * 1024 * 1024},
	}
	fh.SetAccelerators(custom)
	accs2 := fh.Accelerators()
	if len(accs2) != 2 || accs2[1].ID != "cuda:1" {
		t.Errorf("unexpected custom fake accelerators: %+v", accs2)
	}
}

func TestProcessError_Classification(t *testing.T) {
	oomErr := &host.ProcessError{ExitCode: 1, OOM: true, Stderr: "CUDA error: out of memory"}
	if !host.IsOOM(oomErr) {
		t.Errorf("expected IsOOM true")
	}
	if !strings.Contains(oomErr.Error(), "out-of-memory") {
		t.Errorf("unexpected error string: %s", oomErr.Error())
	}

	crashErr := &host.ProcessError{ExitCode: 1, OOM: false, Stderr: "Segmentation fault"}
	if host.IsOOM(crashErr) {
		t.Errorf("expected IsOOM false for crash")
	}
	if !strings.Contains(crashErr.Error(), "crashed") {
		t.Errorf("unexpected error string: %s", crashErr.Error())
	}

	if host.IsOOM(nil) {
		t.Errorf("expected IsOOM false for nil")
	}
}

func launchTestHelper(t *testing.T, mode string) (host.Instance, context.Context) {
	t.Helper()
	t.Setenv("GO_WANT_HELPER_PROCESS", "1")
	t.Setenv("GO_WANT_HELPER_MODE", mode)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)

	rh := host.New()
	argv := []string{os.Args[0], "-test.run=TestHelperProcess", "--", "helper"}
	inst, err := rh.Launch(ctx, argv)
	if err != nil {
		t.Fatalf("Launch failed: %v", err)
	}
	return inst, ctx
}

func TestRealHost_OOMClassification(t *testing.T) {
	inst, _ := launchTestHelper(t, "oom")

	<-inst.Done()
	exitErr := inst.Err()
	if exitErr == nil {
		t.Fatalf("expected non-nil error for OOM helper process")
	}
	if !host.IsOOM(exitErr) {
		t.Errorf("expected exitErr to be classified as OOM, got: %v", exitErr)
	}
}

func TestRealHost_NonMemoryCrashClassification(t *testing.T) {
	inst, _ := launchTestHelper(t, "crash")

	<-inst.Done()
	exitErr := inst.Err()
	if exitErr == nil {
		t.Fatalf("expected non-nil error for crash helper process")
	}
	if host.IsOOM(exitErr) {
		t.Errorf("expected exitErr to NOT be classified as OOM, got: %v", exitErr)
	}
}

func TestRealHost_ObservedAllocation(t *testing.T) {
	inst, ctx := launchTestHelper(t, "alloc")
	t.Cleanup(func() { _ = inst.Stop(context.Background()) })

	if err := inst.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady failed: %v", err)
	}

	// Wait briefly for stderr parsing to process lines
	time.Sleep(100 * time.Millisecond)

	alloc := inst.ObservedAllocation()
	if alloc.VRAM != 4096*1024*1024 {
		t.Errorf("expected VRAM allocation 4096 MiB (%d bytes), got %d", 4096*1024*1024, alloc.VRAM)
	}
	if alloc.RAM <= 0 && os.Getenv("CI") == "" {
		t.Logf("Observed RAM: %d bytes", alloc.RAM)
	}

	_ = inst.Stop(ctx)
}
