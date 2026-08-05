package supervisor_test

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

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
