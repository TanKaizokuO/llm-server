package supervisor_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

func TestRegistry_Tuning_Convergence(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-tuning-model.gguf", "llama", "Q4_K_M")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		var ngl, ctxLen int
		for i, arg := range argv {
			if arg == "-ngl" && i+1 < len(argv) {
				ngl, _ = strconv.Atoi(argv[i+1])
			}
			if arg == "-c" && i+1 < len(argv) {
				ctxLen, _ = strconv.Atoi(argv[i+1])
			}
		}

		// Fake memory rules:
		// Fits if (ctxLen <= 1024 && ngl <= 25)
		if ctxLen > 1024 {
			return nil, &host.ProcessError{OOM: true}
		}
		if ngl > 25 {
			return nil, &host.ProcessError{OOM: true}
		}

		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(h, tmpDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	body := []byte(`{"model":"test-tuning-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK, got %d. Body: %s", rr.Code, rr.Body.String())
	}

	launches := h.Launches()
	if len(launches) == 0 {
		t.Fatal("Expected launches during tuning, got none")
	}
	// Evict the resident instance so we can observe a fresh launch using the cached config
	sup.Evict(context.Background(), "test-tuning-model:q4_k_m")

	// Make a second request to trigger a launch from the tuned cache
	req2 := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr2 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr2, req2)
	if rr2.Code != http.StatusOK {
		t.Fatalf("Expected OK on second request, got %d", rr2.Code)
	}

	launches = h.Launches()
	lastLaunch := launches[len(launches)-1]
	hasCtx1024 := false
	hasNGL25 := false
	for i, arg := range lastLaunch {
		if arg == "-c" && lastLaunch[i+1] == "1024" {
			hasCtx1024 = true
		}
		if arg == "-ngl" && lastLaunch[i+1] == "25" {
			hasNGL25 = true
		}
	}

	if !hasCtx1024 || !hasNGL25 {
		t.Errorf("expected final launch to have -c 1024 and -ngl 25, got %v", lastLaunch)
	}
}

func TestRegistry_Tuning_ContextLadder(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-ladder-model.gguf", "llama", "Q4_K_M")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		var ngl, ctxLen int
		for i, arg := range argv {
			if arg == "-ngl" && i+1 < len(argv) {
				ngl, _ = strconv.Atoi(argv[i+1])
			}
			if arg == "-c" && i+1 < len(argv) {
				ctxLen, _ = strconv.Atoi(argv[i+1])
			}
		}

		// Fake memory rules:
		// Fits only if ctxLen <= 2048 and ngl <= 10
		// The model default ctx is 4096.
		if ctxLen > 2048 {
			return nil, &host.ProcessError{OOM: true}
		}
		if ngl > 10 {
			return nil, &host.ProcessError{OOM: true}
		}

		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(h, tmpDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	body := []byte(`{"model":"test-ladder-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK, got %d", rr.Code)
	}

	launches := h.Launches()
	lastLaunch := launches[len(launches)-1]

	// We expect the final successful launch to have been cached
	sup.Evict(context.Background(), "test-ladder-model:q4_k_m")
	req2 := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr2 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr2, req2)

	launches = h.Launches()
	lastLaunch = launches[len(launches)-1]

	hasCtx2048 := false
	hasNGL10 := false
	for i, arg := range lastLaunch {
		if arg == "-c" && lastLaunch[i+1] == "2048" {
			hasCtx2048 = true
		}
		if arg == "-ngl" && lastLaunch[i+1] == "10" {
			hasNGL10 = true
		}
	}

	if !hasCtx2048 || !hasNGL10 {
		t.Errorf("expected final launch to have -c 2048 and -ngl 10, got %v", lastLaunch)
	}
}

func TestRegistry_Tuning_NonMemoryCrash(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-crash-model.gguf", "llama", "Q4_K_M")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		return nil, &host.ProcessError{OOM: false, ExitCode: 1, Stderr: "segmentation fault"}
	})

	sup, err := supervisor.New(h, tmpDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	body := []byte(`{"model":"test-crash-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("Expected 500 Internal Server Error, got %d", rr.Code)
	}
	if !bytes.Contains(rr.Body.Bytes(), []byte("tuning aborted due to non-memory error")) {
		t.Fatalf("Expected response to surface the crash error, got %s", rr.Body.String())
	}
}
func TestTuning_PersistenceAndReload(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-persist-model.gguf", "llama", "Q4_K_M")
	cachePath := filepath.Join(tmpDir, "tuning.json")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		var ngl int
		for i, arg := range argv {
			if arg == "-ngl" && i+1 < len(argv) {
				ngl, _ = strconv.Atoi(argv[i+1])
			}
		}
		if ngl > 25 {
			return nil, &host.ProcessError{OOM: true, ExitCode: 1, Stderr: "out of memory"}
		}
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup1, err := supervisor.NewWithOpts(h, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts error = %v", err)
	}

	body := []byte(`{"model":"test-persist-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup1.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK, got %d", rr.Code)
	}

	if _, err := os.Stat(cachePath); err != nil {
		t.Fatalf("Expected cache file at %s, got error: %v", cachePath, err)
	}

	launchesBefore := len(h.Launches())

	sup2, err := supervisor.NewWithOpts(h, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts sup2 error = %v", err)
	}

	req2 := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr2 := httptest.NewRecorder()
	sup2.Handler().ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusOK {
		t.Fatalf("Expected OK on reloaded supervisor, got %d", rr2.Code)
	}

	// Should not run new tuning launches; should launch directly using cached config (1 launch for instance launch)
	launchesAfter := len(h.Launches()) - launchesBefore
	if launchesAfter != 1 {
		t.Fatalf("Expected exactly 1 launch on reloaded supervisor, got %d", launchesAfter)
	}
}

func TestTuning_Invalidation_FingerprintChange(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-fp-model.gguf", "llama", "Q4_K_M")
	cachePath := filepath.Join(tmpDir, "tuning.json")

	onLaunch := func(argv []string) (http.Handler, error) {
		var ngl int
		for i, arg := range argv {
			if arg == "-ngl" && i+1 < len(argv) {
				ngl, _ = strconv.Atoi(argv[i+1])
			}
		}
		if ngl > 25 {
			return nil, &host.ProcessError{OOM: true, ExitCode: 1, Stderr: "out of memory"}
		}
		return http.HandlerFunc(host.DefaultMockHandler), nil
	}

	h1 := host.NewFakeHost()
	h1.SetFingerprint("fake-fingerprint-1")
	h1.SetOnLaunch(onLaunch)

	sup1, err := supervisor.NewWithOpts(h1, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts error = %v", err)
	}

	body := []byte(`{"model":"test-fp-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup1.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK, got %d", rr.Code)
	}

	// Now create a supervisor with host fingerprint 2
	h2 := host.NewFakeHost()
	h2.SetFingerprint("fake-fingerprint-2")
	h2.SetOnLaunch(onLaunch)

	sup2, err := supervisor.NewWithOpts(h2, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts sup2 error = %v", err)
	}

	req2 := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr2 := httptest.NewRecorder()
	sup2.Handler().ServeHTTP(rr2, req2)
	if rr2.Code != http.StatusOK {
		t.Fatalf("Expected OK on sup2, got %d", rr2.Code)
	}

	// Fingerprint 2 should trigger a re-tune (multiple launches)
	launches2 := h2.Launches()
	if len(launches2) <= 1 {
		t.Fatalf("Expected re-tuning launches on fingerprint change, got %d launches", len(launches2))
	}
}

func TestTuning_Invalidation_GGUFFileChange(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := writeTestGGUF(t, tmpDir, "test-gguf-change.gguf", "llama", "Q4_K_M")
	cachePath := filepath.Join(tmpDir, "tuning.json")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		var ngl int
		for i, arg := range argv {
			if arg == "-ngl" && i+1 < len(argv) {
				ngl, _ = strconv.Atoi(argv[i+1])
			}
		}
		if ngl > 25 {
			return nil, &host.ProcessError{OOM: true, ExitCode: 1, Stderr: "out of memory"}
		}
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.NewWithOpts(h, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts error = %v", err)
	}

	body := []byte(`{"model":"test-gguf-change:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK, got %d", rr.Code)
	}

	initialLaunches := len(h.Launches())

	// Modify the GGUF file content on disk to alter its size/digest
	if err := os.WriteFile(modelPath, append([]byte("extra content"), []byte("...")...), 0644); err != nil {
		t.Fatalf("modifying model file: %v", err)
	}

	// Evict resident instance to force resolution / launch check
	sup.Evict(context.Background(), "test-gguf-change:q4_k_m")

	req2 := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr2 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr2, req2)

	// Since digest changed, cache entry should be invalidated and re-tuning triggered
	launchesAfter := len(h.Launches()) - initialLaunches
	if launchesAfter <= 1 {
		t.Fatalf("Expected re-tuning on GGUF change, got %d launches", launchesAfter)
	}
}

func TestTuning_NativeEndpoints(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-native-model.gguf", "llama", "Q4_K_M")
	cachePath := filepath.Join(tmpDir, "tuning.json")

	h := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(h, []string{tmpDir}, supervisor.WithCachePath(cachePath))
	if err != nil {
		t.Fatalf("NewWithOpts error = %v", err)
	}

	// 1. GET /v1/tuning initial (idle, empty entries)
	reqGet := httptest.NewRequest("GET", "/v1/tuning", nil)
	rrGet := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet, reqGet)
	if rrGet.Code != http.StatusOK {
		t.Fatalf("GET /v1/tuning status = %d", rrGet.Code)
	}
	if !bytes.Contains(rrGet.Body.Bytes(), []byte(`"status":"idle"`)) {
		t.Errorf("Expected status idle, got %s", rrGet.Body.String())
	}

	// 2. Trigger tuning
	body := []byte(`{"model":"test-native-model:q4_k_m"}`)
	reqChat := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rrChat := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrChat, reqChat)
	if rrChat.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status = %d", rrChat.Code)
	}

	// 3. GET /v1/tuning after tuning (contains entry for test-native-model:q4_k_m)
	rrGet2 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet2, reqGet)
	if rrGet2.Code != http.StatusOK {
		t.Fatalf("GET /v1/tuning status = %d", rrGet2.Code)
	}
	if !bytes.Contains(rrGet2.Body.Bytes(), []byte("test-native-model:q4_k_m")) {
		t.Errorf("Expected tuning entry for model, got %s", rrGet2.Body.String())
	}

	// 4. POST /v1/tuning/reset with model query param
	reqReset := httptest.NewRequest("POST", "/v1/tuning/reset?model=test-native-model:q4_k_m", nil)
	rrReset := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrReset, reqReset)
	if rrReset.Code != http.StatusOK {
		t.Fatalf("POST /v1/tuning/reset status = %d", rrReset.Code)
	}

	// 5. GET /v1/tuning after reset should be empty again
	rrGet3 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet3, reqGet)
	if !bytes.Contains(rrGet3.Body.Bytes(), []byte(`"entries":{}`)) {
		t.Errorf("Expected entries to be empty after reset, got %s", rrGet3.Body.String())
	}
}

func TestTuning_BudgetExceeded_FallbackToConservativeConfig(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-budget-model.gguf", "llama", "Q4_K_M")
	cachePath := filepath.Join(tmpDir, "tuning.json")

	h := host.NewFakeHost()
	// Set launch hook to delay probes so budget expires during tuning
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		time.Sleep(50 * time.Millisecond)
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	// Configure supervisor with a tight 20ms tuning budget
	sup, err := supervisor.NewWithOpts(h, []string{tmpDir},
		supervisor.WithCachePath(cachePath),
		supervisor.WithTuningBudget(20*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("NewWithOpts error = %v", err)
	}

	body := []byte(`{"model":"test-budget-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected OK response after fallback, got %d. Body: %s", rr.Code, rr.Body.String())
	}

	// Verify that the final launch used conservative fallback configuration (-ngl 0)
	lastLaunch := h.LastLaunch()
	if lastLaunch == nil {
		t.Fatal("Expected launch, got none")
	}
	hasNGL0 := false
	for i, arg := range lastLaunch {
		if arg == "-ngl" || arg == "--n-gpu-layers" {
			if i+1 < len(lastLaunch) && lastLaunch[i+1] == "0" {
				hasNGL0 = true
			}
		}
	}
	if !hasNGL0 {
		t.Errorf("Expected fallback launch to use offload 0, got argv %v", lastLaunch)
	}

	// Verify native GET /v1/tuning reflects the fallback entry (Offload == 0)
	reqGet := httptest.NewRequest("GET", "/v1/tuning", nil)
	rrGet := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet, reqGet)
	if !bytes.Contains(rrGet.Body.Bytes(), []byte(`"offload":0`)) {
		t.Errorf("Expected tuning entry offload to be 0, got %s", rrGet.Body.String())
	}
}

func TestTuning_ProgressObservableOnNativeEndpoint(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-progress-model.gguf", "llama", "Q4_K_M")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		time.Sleep(100 * time.Millisecond)
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(h, tmpDir)
	if err != nil {
		t.Fatalf("New error = %v", err)
	}

	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		body := []byte(`{"model":"test-progress-model:q4_k_m"}`)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
		rr := httptest.NewRecorder()
		sup.Handler().ServeHTTP(rr, req)
	}()

	// Wait briefly for tuning to begin
	time.Sleep(30 * time.Millisecond)

	// Issue GET /v1/tuning while tuning is in progress
	reqGet := httptest.NewRequest("GET", "/v1/tuning", nil)
	rrGet := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet, reqGet)

	if rrGet.Code != http.StatusOK {
		t.Fatalf("GET /v1/tuning returned status %d", rrGet.Code)
	}

	var resp struct {
		Status      string `json:"status"`
		ActiveModel string `json:"active_model"`
		Progress    *struct {
			StartedAt      string  `json:"started_at"`
			ElapsedSeconds float64 `json:"elapsed_seconds"`
			CurrentCtx     uint64  `json:"current_ctx"`
			CurrentOffload uint64  `json:"current_offload"`
			ProbeCount     int     `json:"probe_count"`
		} `json:"progress"`
	}

	if err := json.Unmarshal(rrGet.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Decoding /v1/tuning response: %v", err)
	}

	if resp.Status != "tuning" {
		t.Errorf("Expected status 'tuning', got %q", resp.Status)
	}
	if !strings.Contains(resp.ActiveModel, "test-progress-model") {
		t.Errorf("Expected active_model to contain test-progress-model, got %q", resp.ActiveModel)
	}
	if resp.Progress == nil {
		t.Fatal("Expected progress object while tuning, got nil")
	}
	if resp.Progress.ProbeCount < 1 {
		t.Errorf("Expected probe_count >= 1, got %d", resp.Progress.ProbeCount)
	}

	<-doneCh

	// After completion, status should revert to idle and progress to nil
	rrGet2 := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet2, reqGet)
	if bytes.Contains(rrGet2.Body.Bytes(), []byte(`"status":"tuning"`)) {
		t.Errorf("Expected status idle after completion, got %s", rrGet2.Body.String())
	}
}

func TestTuning_ClientDisconnectDoesNotTriggerFallback(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "test-cancel-model.gguf", "llama", "Q4_K_M")

	h := host.NewFakeHost()
	h.SetOnLaunch(func(argv []string) (http.Handler, error) {
		time.Sleep(100 * time.Millisecond)
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(h, tmpDir)
	if err != nil {
		t.Fatalf("New error = %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	body := []byte(`{"model":"test-cancel-model:q4_k_m"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body)).WithContext(ctx)
	rr := httptest.NewRecorder()

	sup.Handler().ServeHTTP(rr, req)

	// Since client context timed out/cancelled, no tuned entry should be recorded
	reqGet := httptest.NewRequest("GET", "/v1/tuning", nil)
	rrGet := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrGet, reqGet)

	if bytes.Contains(rrGet.Body.Bytes(), []byte("test-cancel-model")) {
		t.Errorf("Expected no tuning entry after client cancellation, got %s", rrGet.Body.String())
	}
}
