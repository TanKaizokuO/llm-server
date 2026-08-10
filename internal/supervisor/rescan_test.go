package supervisor_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
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

// TestRescan_DisabledByZeroInterval ensures a rescanInterval of 0 turns off
// the background timer, without breaking on-demand rescanning.
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

	if err := sup.Rescan(context.Background()); err != nil {
		t.Fatalf("Rescan failed: %v", err)
	}
	if ids := listedModelIDs(t, sup); len(ids) != 2 {
		t.Fatalf("models after explicit Rescan = %v, want 2", ids)
	}
}

// TestRescan_DeletedModelEvictsInstance ensures a scan that comes back
// without a previously served model will remove it from the list and
// evict any running instance of it.
func TestRescan_DeletedModelEvictsInstance(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := writeTestGGUF(t, tmpDir, "only-model.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	if ids := listedModelIDs(t, sup); len(ids) != 1 {
		t.Fatalf("initial models = %v, want 1", ids)
	}

	// Start a generation to make the instance resident
	body := []byte(`{"model":"only-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Generate status = %d, want 200", rr.Code)
	}

	instances := fakeHost.Instances()
	if len(instances) != 1 {
		t.Fatalf("expected 1 running instance, got %d", len(instances))
	}

	// Now delete the model file
	if err := os.Remove(modelPath); err != nil {
		t.Fatalf("removing model file: %v", err)
	}

	// Rescan should discover the file is gone, remove the model, and evict the instance
	if err := sup.Rescan(context.Background()); err != nil {
		t.Fatalf("Rescan failed: %v", err)
	}

	if ids := listedModelIDs(t, sup); len(ids) != 0 {
		t.Fatalf("models after removing the file = %v, want 0", ids)
	}

	// The instance must be gone from the registry, not merely marked stopped;
	// /api/ps is the observable surface that proves it.
	reqPs := httptest.NewRequest("GET", "/api/ps", nil)
	rrPs := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rrPs, reqPs)
	if rrPs.Code != http.StatusOK {
		t.Fatalf("GET /api/ps status = %d", rrPs.Code)
	}
	if string(rrPs.Body.Bytes()) != `{"models":[]}`+"\n" {
		t.Errorf("GET /api/ps = %q, want %q", string(rrPs.Body.Bytes()), `{"models":[]}`+"\n")
	}
}

func TestRescan_DeletedModelDrainsInFlightRequest(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := writeTestGGUF(t, tmpDir, "only-model.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	slowHandlerStarted := make(chan struct{})
	slowHandlerRelease := make(chan struct{})

	fakeHost.SetOnLaunch(func(argv []string) (http.Handler, error) {
		h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/chat/completions" {
				close(slowHandlerStarted)
				<-slowHandlerRelease
			}
			host.DefaultMockHandler(w, r)
		})
		return h, nil
	})

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	respCh := make(chan *http.Response, 1)

	go func() {
		body := []byte(`{"model":"only-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
		resp, _ := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		respCh <- resp
	}()

	<-slowHandlerStarted

	if err := os.Remove(modelPath); err != nil {
		t.Fatalf("removing model file: %v", err)
	}

	rescanDone := make(chan struct{})
	go func() {
		_ = sup.Rescan(context.Background())
		close(rescanDone)
	}()

	// Ensure Rescan has blocked in evictByID
	time.Sleep(50 * time.Millisecond)

	// Unblock request
	close(slowHandlerRelease)

	resp := <-respCh
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Generate status = %d, want 200 (drain failed to complete successfully)", resp.StatusCode)
	}

	<-rescanDone
}

// TestRescan_StartsWithNoModels_OnDemand ensures a supervisor can start
// with an empty directory and pick up a model dropped in later.
func TestRescan_StartsWithNoModels_OnDemand(t *testing.T) {
	tmpDir := t.TempDir()
	sup, err := supervisor.New(host.NewFakeHost(), tmpDir)
	if err != nil {
		t.Fatalf("New failed on empty dir: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	if ids := listedModelIDs(t, sup); len(ids) != 0 {
		t.Fatalf("initial models = %v, want 0", ids)
	}

	writeTestGGUF(t, tmpDir, "new-model.gguf", "llama", "Q4_K_M")

	if err := sup.Rescan(context.Background()); err != nil {
		t.Fatalf("Rescan failed: %v", err)
	}

	if ids := listedModelIDs(t, sup); len(ids) != 1 {
		t.Fatalf("models after scan = %v, want 1", ids)
	}
}

// TestRescan_StartsWithNoModels_Timer ensures the background timer works
// correctly from a zero-model start.
func TestRescan_StartsWithNoModels_Timer(t *testing.T) {
	tmpDir := t.TempDir()

	fake := host.NewFakeHost()
	// Short rescan interval
	sup, err := supervisor.NewWithOpts(fake, []string{tmpDir}, supervisor.WithRescanInterval(10*time.Millisecond))
	if err != nil {
		t.Fatalf("NewWithOpts failed on empty dir: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	if ids := listedModelIDs(t, sup); len(ids) != 0 {
		t.Fatalf("initial models = %v, want 0", ids)
	}

	writeTestGGUF(t, tmpDir, "new-model.gguf", "llama", "Q4_K_M")

	// Wait for the background timer to discover it
	deadline := time.Now().Add(time.Second)
	var found []string
	for time.Now().Before(deadline) {
		found = listedModelIDs(t, sup)
		if len(found) == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if len(found) != 1 {
		t.Fatalf("models after waiting for timer = %v, want 1", found)
	}
}

func TestRescan_ClosePromptly(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "model.gguf", "llama", "Q4_K_M")

	fake := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fake, []string{tmpDir}, supervisor.WithRescanInterval(time.Millisecond))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}

	// Wait long enough for at least one tick to fire
	time.Sleep(10 * time.Millisecond)

	start := time.Now()
	err = sup.Close()
	dur := time.Since(start)

	if err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
	if dur > time.Second {
		t.Errorf("Close() took too long: %v (expected <1s)", dur)
	}
}

func TestRescan_AfterClose(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "model.gguf", "llama", "Q4_K_M")

	fake := host.NewFakeHost()
	sup, err := supervisor.New(fake, tmpDir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if err := sup.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	err = sup.Rescan(context.Background())
	if err == nil {
		t.Fatal("expected error calling Rescan after Close, got nil")
	}
	if err.Error() != "supervisor is closed" {
		t.Errorf("expected 'supervisor is closed' error, got: %v", err)
	}
}
