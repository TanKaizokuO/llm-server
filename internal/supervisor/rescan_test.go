package supervisor_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

func listedModelIDs(t *testing.T, sup *supervisor.Supervisor) []string {
	t.Helper()
	req := httptest.NewRequest("GET", "/api/tags", nil)
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /api/tags status = %d", rr.Code)
	}
	var resp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decoding /api/tags: %v", err)
	}
	ids := make([]string, len(resp.Models))
	for i, m := range resp.Models {
		ids[i] = m.Name
	}
	return ids
}

// TestRescan_OnDemand covers: "Directories are rescanned on demand ... a
// newly dropped GGUF becomes servable without a restart."
func TestRescan_OnDemand(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "first-model.gguf", "llama", "Q4_K_M")

	sup, err := supervisor.New(host.NewFakeHost(), tmpDir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	if ids := listedModelIDs(t, sup); len(ids) != 1 {
		t.Fatalf("initial models = %v, want 1", ids)
	}

	writeTestGGUF(t, tmpDir, "second-model.gguf", "llama", "Q4_K_M")

	req := httptest.NewRequest("POST", "/v1/rescan", nil)
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("POST /v1/rescan status = %d, body: %s", rr.Code, rr.Body.String())
	}

	ids := listedModelIDs(t, sup)
	if len(ids) != 2 {
		t.Fatalf("models after rescan = %v, want 2", ids)
	}
}

// TestRescan_Timer covers: "Directories are rescanned ... on a timer; a
// newly dropped GGUF becomes servable without a restart" — with no explicit
// Rescan call, using a short interval against the real clock.
func TestRescan_Timer(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "first-model.gguf", "llama", "Q4_K_M")

	sup, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithRescanInterval(20*time.Millisecond))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	writeTestGGUF(t, tmpDir, "second-model.gguf", "llama", "Q4_K_M")

	deadline := time.Now().Add(2 * time.Second)
	for {
		if len(listedModelIDs(t, sup)) == 2 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for periodic rescan to discover the new model")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestRescan_DisabledByDefaultInterval ensures a rescanInterval of 0 turns
// off the background timer without breaking on-demand rescanning.
func TestRescan_DisabledByZeroInterval(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "first-model.gguf", "llama", "Q4_K_M")

	sup, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithRescanInterval(0))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	writeTestGGUF(t, tmpDir, "second-model.gguf", "llama", "Q4_K_M")

	time.Sleep(100 * time.Millisecond)
	if ids := listedModelIDs(t, sup); len(ids) != 1 {
		t.Fatalf("models with timer disabled = %v, want 1 (no periodic rescan should have run)", ids)
	}

	if err := sup.Rescan(); err != nil {
		t.Fatalf("Rescan failed: %v", err)
	}
	if ids := listedModelIDs(t, sup); len(ids) != 2 {
		t.Fatalf("models after explicit Rescan = %v, want 2", ids)
	}
}
