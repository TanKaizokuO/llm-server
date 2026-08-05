package supervisor_test

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
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