package supervisor_test

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

func writeTestGGUF(t *testing.T, dir, filename, arch, quant string, extraKV ...map[string]any) string {
	t.Helper()
	path := filepath.Join(dir, filename)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("creating dir for %s: %v", filename, err)
	}

	var kv map[string]any
	if len(extraKV) > 0 {
		kv = extraKV[0]
	}

	data := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  arch,
		ContextLength: 4096,
		Quantization:  quant,
		ExtraKV:       kv,
	})
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("writing GGUF %s: %v", filename, err)
	}
	return path
}

func newTestServer(t *testing.T, dirs ...string) (*httptest.Server, *host.FakeHost) {
	t.Helper()
	fakeHost := host.NewFakeHost()
	if len(dirs) == 0 {
		tmpDir := t.TempDir()
		writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
		dirs = []string{tmpDir}
	}
	s, err := supervisor.New(fakeHost, dirs...)
	if err != nil {
		t.Fatalf("supervisor.New failed: %v", err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(func() {
		srv.Close()
		_ = s.Close()
	})
	return srv, fakeHost
}

func TestHealthReportsReadyWithNoModelLoaded(t *testing.T) {
	srv, _ := newTestServer(t)

	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	var got struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got.Status != "ready" {
		t.Errorf("status field = %q, want %q", got.Status, "ready")
	}
}

func TestDiscovery_RefusesToStartWhenNoModelsFound(t *testing.T) {
	tmpDir := t.TempDir()

	_, err := supervisor.New(host.NewFakeHost(), tmpDir)
	if err == nil {
		t.Fatal("expected supervisor.New to fail on empty directory, got nil")
	}

	if !strings.Contains(err.Error(), "no models found") {
		t.Errorf("error message = %q, want containing 'no models found'", err.Error())
	}
}

func TestDiscovery_RecursiveScanShardsProjectorsCorruptFiles(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Root level model
	writeTestGGUF(t, tmpDir, "model1-q4_k_m.gguf", "llama", "Q4_K_M")

	// 2. Nested directory model
	writeTestGGUF(t, tmpDir, "nested/deep/model2-q8_0.gguf", "mistral", "Q8_0")

	// 3. Multi-part sharded model (3 shards by filename)
	shard1Path := writeTestGGUF(t, tmpDir, "sharded-q4_k_m-00001-of-00003.gguf", "gemma", "Q4_K_M")
	writeTestGGUF(t, tmpDir, "sharded-q4_k_m-00002-of-00003.gguf", "gemma", "Q4_K_M")
	writeTestGGUF(t, tmpDir, "sharded-q4_k_m-00003-of-00003.gguf", "gemma", "Q4_K_M")

	// 4. Pure metadata model (no quant token in filename)
	writeTestGGUF(t, tmpDir, "puremodel.gguf", "qwen2", "Q4_K_M")

	// 5. Sharded model by GGUF metadata split.no key
	writeTestGGUF(t, tmpDir, "metashard-00001-of-00002.gguf", "llama", "Q4_K_M", map[string]any{"split.no": uint16(0)})
	writeTestGGUF(t, tmpDir, "metashard-00002-of-00002.gguf", "llama", "Q4_K_M", map[string]any{"split.no": uint16(1)})

	// 6. Projector files (should be excluded)
	writeTestGGUF(t, tmpDir, "mmproj-model-f16.gguf", "llama", "F16")
	writeTestGGUF(t, tmpDir, "clip-vision-proj.gguf", "clip", "F16")

	// 7. Corrupt file (invalid magic)
	corruptPath := filepath.Join(tmpDir, "corrupt.gguf")
	if err := os.WriteFile(corruptPath, []byte("NOT_A_VALID_GGUF_HEADER_BYTES"), 0644); err != nil {
		t.Fatalf("writing corrupt gguf: %v", err)
	}

	// 8. Truncated file (valid magic, truncated header)
	truncatedPath := filepath.Join(tmpDir, "truncated.gguf")
	if err := os.WriteFile(truncatedPath, []byte{'G', 'G', 'U', 'F', 0x03, 0x00, 0x00, 0x00}, 0644); err != nil {
		t.Fatalf("writing truncated gguf: %v", err)
	}

	srv, _ := newTestServer(t, tmpDir)

	// --- Assert Ollama surface: GET /api/tags ---
	respTags, err := http.Get(srv.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer respTags.Body.Close()

	if respTags.StatusCode != http.StatusOK {
		t.Errorf("GET /api/tags status = %d, want %d", respTags.StatusCode, http.StatusOK)
	}

	var tags struct {
		Models []struct {
			Name    string `json:"name"`
			Model   string `json:"model"`
			Size    int64  `json:"size"`
			Digest  string `json:"digest"`
			Details struct {
				Format            string `json:"format"`
				Family            string `json:"family"`
				QuantizationLevel string `json:"quantization_level"`
			} `json:"details"`
		} `json:"models"`
	}

	if err := json.NewDecoder(respTags.Body).Decode(&tags); err != nil {
		t.Fatalf("decoding /api/tags: %v", err)
	}

	wantModels := map[string]struct {
		family string
		quant  string
	}{
		"metashard:q4_k_m": {family: "llama", quant: "Q4_K_M"},
		"model1:q4_k_m":    {family: "llama", quant: "Q4_K_M"},
		"model2:q8_0":      {family: "mistral", quant: "Q8_0"},
		"puremodel:q4_k_m": {family: "qwen2", quant: "Q4_K_M"},
		"sharded:q4_k_m":   {family: "gemma", quant: "Q4_K_M"},
	}

	if len(tags.Models) != len(wantModels) {
		t.Fatalf("got %d models in /api/tags, want %d", len(tags.Models), len(wantModels))
	}

	for _, m := range tags.Models {
		expected, ok := wantModels[m.Name]
		if !ok {
			t.Errorf("unexpected model in /api/tags: %s", m.Name)
			continue
		}
		if m.Model != m.Name {
			t.Errorf("model field = %q, want %q", m.Model, m.Name)
		}
		if m.Details.Format != "gguf" {
			t.Errorf("format = %q, want 'gguf'", m.Details.Format)
		}
		if m.Details.Family != expected.family {
			t.Errorf("family for %s = %q, want %q", m.Name, m.Details.Family, expected.family)
		}
		if m.Details.QuantizationLevel != expected.quant {
			t.Errorf("quantization for %s = %q, want %q", m.Name, m.Details.QuantizationLevel, expected.quant)
		}
		if !strings.HasPrefix(m.Digest, "sha256:") {
			t.Errorf("digest for %s = %q, want prefix 'sha256:'", m.Name, m.Digest)
		}
	}

	// Verify sharded model path points to 1st shard
	_ = shard1Path

	// --- Assert OpenAI surface: GET /v1/models ---
	respV1, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer respV1.Body.Close()

	if respV1.StatusCode != http.StatusOK {
		t.Errorf("GET /v1/models status = %d, want %d", respV1.StatusCode, http.StatusOK)
	}

	var v1Models struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.NewDecoder(respV1.Body).Decode(&v1Models); err != nil {
		t.Fatalf("decoding /v1/models: %v", err)
	}

	if v1Models.Object != "list" {
		t.Errorf("object = %q, want 'list'", v1Models.Object)
	}

	if len(v1Models.Data) != len(wantModels) {
		t.Fatalf("got %d models in /v1/models, want %d", len(v1Models.Data), len(wantModels))
	}

	for i, m := range v1Models.Data {
		if _, ok := wantModels[m.ID]; !ok {
			t.Errorf("unexpected model in /v1/models: %s", m.ID)
		}
		if m.Object != "model" {
			t.Errorf("data[%d].object = %q, want 'model'", i, m.Object)
		}
		if m.OwnedBy != "llm-server" {
			t.Errorf("data[%d].owned_by = %q, want 'llm-server'", i, m.OwnedBy)
		}

		// Ensure Ollama and OpenAI surfaces report identical model IDs in order
		if tags.Models[i].Name != m.ID {
			t.Errorf("model index %d mismatch: Ollama=%s, OpenAI=%s", i, tags.Models[i].Name, m.ID)
		}
	}
}

func TestDiscovery_MultipleDirectories(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()

	writeTestGGUF(t, dir1, "modelA.q4_k_m.gguf", "llama", "Q4_K_M")
	writeTestGGUF(t, dir2, "modelB.q8_0.gguf", "mistral", "Q8_0")

	srv, _ := newTestServer(t, dir1, dir2)

	resp, err := http.Get(srv.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()

	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(tags.Models) != 2 {
		t.Fatalf("got %d models, want 2", len(tags.Models))
	}
	if tags.Models[0].Name != "modela:q4_k_m" || tags.Models[1].Name != "modelb:q8_0" {
		t.Errorf("got models %v, want ['modela:q4_k_m', 'modelb:q8_0']", tags.Models)
	}
}

func TestV1ChatCompletions_StreamsTokensAndAssertsArgv(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
	srv, fakeHost := newTestServer(t, tmpDir)

	body := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	req, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want starting with text/event-stream", ct)
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
		t.Errorf("line 0 = %q, want containing 'Hello'", lines[0])
	}
	if !strings.Contains(lines[3], `[DONE]`) {
		t.Errorf("line 3 = %q, want containing '[DONE]'", lines[3])
	}

	launches := fakeHost.Launches()
	if len(launches) != 1 {
		t.Fatalf("expected 1 Host launch, got %d", len(launches))
	}
	wantArgv := []string{"llama-server", "-m", modelPath, "-c", "4096", "-ngl", "100", "-np", "1"}
	gotArgv := launches[0]
	if strings.Join(gotArgv, " ") != strings.Join(wantArgv, " ") {
		t.Errorf("launched argv = %v, want %v", gotArgv, wantArgv)
	}
}

func TestV1ChatCompletions_UnknownModelReturnsOpenAINotFoundError(t *testing.T) {
	srv, _ := newTestServer(t)

	body := `{"model":"non-existent-model:tag","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusNotFound)
	}

	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Param   string `json:"param"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		t.Fatalf("decoding error response: %v", err)
	}

	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("error.type = %q, want %q", errResp.Error.Type, "invalid_request_error")
	}
	if errResp.Error.Param != "model" {
		t.Errorf("error.param = %q, want %q", errResp.Error.Param, "model")
	}
	if errResp.Error.Code != "model_not_found" {
		t.Errorf("error.code = %q, want %q", errResp.Error.Code, "model_not_found")
	}
	if !strings.Contains(errResp.Error.Message, "non-existent-model:tag") {
		t.Errorf("error.message = %q, want containing model name", errResp.Error.Message)
	}
}

func TestV1ChatCompletions_RequestCancellationPropagatesToInstance(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
	fakeHost := host.NewFakeHost()

	cancelledCh := make(chan struct{}, 1)
	fakeHost.SetOnLaunch(func(argv []string) (http.Handler, error) {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			_, _ = w.Write([]byte("data: chunk1\n\n"))
			flusher.Flush()

			<-r.Context().Done()
			select {
			case cancelledCh <- struct{}{}:
			default:
			}
		}), nil
	})

	s, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New failed: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)

	reqCtx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(`{"model":"llama-3-8b:q4_k_m"}`))
	if err != nil {
		t.Fatalf("creating request: %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("doing request: %v", err)
	}

	buf := make([]byte, 64)
	n, _ := resp.Body.Read(buf)
	if !strings.Contains(string(buf[:n]), "chunk1") {
		t.Errorf("read chunk = %q, want containing chunk1", string(buf[:n]))
	}

	cancel()
	_ = resp.Body.Close()

	select {
	case <-cancelledCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for instance context cancellation")
	}
}

func TestModelResolution_ExactNameTagResolvesWhenMultipleQuantisationsExist(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
	pathQ8 := writeTestGGUF(t, tmpDir, "llama-3-8b.q8_0.gguf", "llama", "Q8_0")

	srv, fakeHost := newTestServer(t, tmpDir)

	body := `{"model":"llama-3-8b:q8_0","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	launches := fakeHost.Launches()
	if len(launches) != 1 {
		t.Fatalf("expected 1 Host launch, got %d", len(launches))
	}
	wantArgv := []string{"llama-server", "-m", pathQ8, "-c", "4096", "-ngl", "100", "-np", "1"}
	gotArgv := launches[0]
	if strings.Join(gotArgv, " ") != strings.Join(wantArgv, " ") {
		t.Errorf("launched argv = %v, want %v", gotArgv, wantArgv)
	}
}

func TestModelResolution_BareNameResolvesWhenUnambiguous(t *testing.T) {
	tmpDir := t.TempDir()
	pathQ4 := writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	srv, fakeHost := newTestServer(t, tmpDir)

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	launches := fakeHost.Launches()
	if len(launches) != 1 {
		t.Fatalf("expected 1 Host launch, got %d", len(launches))
	}
	wantArgv := []string{"llama-server", "-m", pathQ4, "-c", "4096", "-ngl", "100", "-np", "1"}
	gotArgv := launches[0]
	if strings.Join(gotArgv, " ") != strings.Join(wantArgv, " ") {
		t.Errorf("launched argv = %v, want %v", gotArgv, wantArgv)
	}
}

func TestModelResolution_BareNameReturnsErrorWhenAmbiguous(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
	writeTestGGUF(t, tmpDir, "llama-3-8b.q8_0.gguf", "llama", "Q8_0")

	srv, fakeHost := newTestServer(t, tmpDir)

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}

	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Param   string `json:"param"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		t.Fatalf("decoding error response: %v", err)
	}

	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("error.type = %q, want %q", errResp.Error.Type, "invalid_request_error")
	}
	if errResp.Error.Param != "model" {
		t.Errorf("error.param = %q, want %q", errResp.Error.Param, "model")
	}
	if errResp.Error.Code != "model_ambiguous" {
		t.Errorf("error.code = %q, want %q", errResp.Error.Code, "model_ambiguous")
	}
	if !strings.Contains(errResp.Error.Message, "llama-3-8b") {
		t.Errorf("error.message = %q, want containing model name 'llama-3-8b'", errResp.Error.Message)
	}
	if !strings.Contains(errResp.Error.Message, "q4_k_m") || !strings.Contains(errResp.Error.Message, "q8_0") {
		t.Errorf("error.message = %q, want containing tags 'q4_k_m' and 'q8_0'", errResp.Error.Message)
	}

	if len(fakeHost.Launches()) != 0 {
		t.Errorf("expected 0 Host launches for ambiguous model reference, got %d", len(fakeHost.Launches()))
	}
}

func TestRegistry_LoadCoalescence(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	slowLaunchStarted := make(chan struct{})
	slowLaunchRelease := make(chan struct{})
	var once sync.Once

	fakeHost.SetOnLaunch(func(argv []string) (http.Handler, error) {
		once.Do(func() {
			close(slowLaunchStarted)
		})
		<-slowLaunchRelease
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	const numGoroutines = 10
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
			resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
			if err != nil {
				errCh <- err
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				errCh <- fmt.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
				return
			}
		}()
	}

	<-slowLaunchStarted
	close(slowLaunchRelease)

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent request error: %v", err)
	}

	if len(fakeHost.Launches()) != 1 {
		t.Errorf("expected exactly 1 Host launch for 10 coalesced requests, got %d", len(fakeHost.Launches()))
	}
}
func TestRegistry_ResidentModelServedWithoutRelaunching(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`

	// First request: loads model
	resp1, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	resp1.Body.Close()
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first status = %d, want %d", resp1.StatusCode, http.StatusOK)
	}

	if len(fakeHost.Launches()) != 1 {
		t.Fatalf("expected 1 launch after first request, got %d", len(fakeHost.Launches()))
	}

	// Second request: served from resident instance
	resp2, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("second status = %d, want %d", resp2.StatusCode, http.StatusOK)
	}

	if len(fakeHost.Launches()) != 1 {
		t.Errorf("expected resident model request to produce 0 extra launches, got total %d", len(fakeHost.Launches()))
	}
}

func TestRegistry_DeadInstanceReapedAndRestarted(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`

	// 1. Initial request loads model
	resp1, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	resp1.Body.Close()

	insts := fakeHost.Instances()
	if len(insts) != 1 {
		t.Fatalf("expected 1 instance launched, got %d", len(insts))
	}

	// 2. Kill the running instance externally (simulate crash/unexpected exit)
	stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := insts[0].Stop(stopCtx); err != nil {
		t.Fatalf("stopping instance: %v", err)
	}

	// 3. Next request reaps dead instance and launches fresh one
	resp2, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("second status = %d, want %d", resp2.StatusCode, http.StatusOK)
	}

	if len(fakeHost.Launches()) != 2 {
		t.Errorf("expected 2 launches after reaping dead instance, got %d", len(fakeHost.Launches()))
	}
}

func TestRegistry_CloseStopsAllInstances(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp1, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	resp1.Body.Close()

	insts := fakeHost.Instances()
	if len(insts) != 1 {
		t.Fatalf("expected 1 instance launched, got %d", len(insts))
	}

	if err := sup.Close(); err != nil {
		t.Fatalf("sup.Close: %v", err)
	}

	// Verify the instance's Done channel is closed (stopped)
	select {
	case <-insts[0].Done():
		// OK, stopped cleanly
	case <-time.After(1 * time.Second):
		t.Errorf("expected instance to be stopped on Supervisor.Close()")
	}

	// Verify new requests after Close fail cleanly
	resp2, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post-close POST: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("post-close status = %d, want %d", resp2.StatusCode, http.StatusServiceUnavailable)
	}
}

func TestRegistry_EvictStopsAndRemovesInstance(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp1, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	resp1.Body.Close()

	insts := fakeHost.Instances()
	if len(insts) != 1 {
		t.Fatalf("expected 1 instance launched, got %d", len(insts))
	}

	// Evict the model
	if err := sup.Evict(context.Background(), "llama-3-8b"); err != nil {
		t.Fatalf("sup.Evict: %v", err)
	}

	// Verify instance stopped
	select {
	case <-insts[0].Done():
		// OK
	case <-time.After(1 * time.Second):
		t.Errorf("expected instance to be stopped after Evict")
	}

	// Next request triggers fresh launch
	resp2, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	resp2.Body.Close()

	if len(fakeHost.Launches()) != 2 {
		t.Errorf("expected 2 launches after Evict, got %d", len(fakeHost.Launches()))
	}

	// Evicting a non-resident model should return nil without error
	if err := sup.Evict(context.Background(), "non-existent-model"); err == nil {
		t.Errorf("Evict(non-existent) error = nil, want error")
	}
}

func TestRegistry_EvictWhileLoading(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()

	launchStarted := make(chan struct{})
	launchRelease := make(chan struct{})
	var once sync.Once

	fakeHost.SetOnLaunch(func(argv []string) (http.Handler, error) {
		once.Do(func() {
			close(launchStarted)
		})
		<-launchRelease
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	// Trigger load in background
	go func() {
		body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
		resp, _ := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if resp != nil {
			resp.Body.Close()
		}
	}()

	<-launchStarted

	// Call Evict while load is in-flight
	evictDone := make(chan error, 1)
	go func() {
		evictDone <- sup.Evict(context.Background(), "llama-3-8b")
	}()

	// Release launch so it completes
	close(launchRelease)

	if err := <-evictDone; err != nil {
		t.Errorf("Evict while loading error = %v, want nil", err)
	}

	// Check that the loaded instance was stopped
	insts := fakeHost.Instances()
	if len(insts) == 1 {
		select {
		case <-insts[0].Done():
			// OK, stopped
		case <-time.After(1 * time.Second):
			t.Errorf("expected instance loaded during Evict to be stopped")
		}
	}
}

func TestRegistry_CloseWhileLoading(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()

	launchStarted := make(chan struct{})
	launchRelease := make(chan struct{})
	var once sync.Once

	fakeHost.SetOnLaunch(func(argv []string) (http.Handler, error) {
		once.Do(func() {
			close(launchStarted)
		})
		<-launchRelease
		return http.HandlerFunc(host.DefaultMockHandler), nil
	})

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	// Trigger load in background
	go func() {
		body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
		resp, _ := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if resp != nil {
			resp.Body.Close()
		}
	}()

	<-launchStarted

	// Close supervisor while load is in-flight
	closeDone := make(chan error, 1)
	go func() {
		closeDone <- sup.Close()
	}()

	// Release launch
	close(launchRelease)

	if err := <-closeDone; err != nil {
		t.Errorf("Close while loading error = %v, want nil", err)
	}

	// Check that the launched instance was stopped
	insts := fakeHost.Instances()
	if len(insts) == 1 {
		select {
		case <-insts[0].Done():
			// OK, stopped
		case <-time.After(1 * time.Second):
			t.Errorf("expected in-flight instance to be stopped on Close()")
		}
	}
}
func TestTTL_IdleInstanceStopsAfterTTL(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithDefaultTTL(50*time.Millisecond))
	if err != nil {
		t.Fatalf("supervisor.NewWithOpts: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	insts := fakeHost.Instances()
	if len(insts) == 0 {
		t.Fatalf("expected 1 instance launched, got 0")
	}
	inst := insts[len(insts)-1]

	select {
	case <-inst.Done():
		t.Fatal("instance stopped immediately before TTL elapsed")
	default:
	}

	select {
	case <-inst.Done():
		// OK, stopped after TTL
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected instance to be stopped after 50ms TTL")
	}
}

func TestTTL_PerModelTTLOverridesGlobalDefault(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir},
		supervisor.WithDefaultTTL(5*time.Second),
		supervisor.WithModelTTL("llama-3-8b.q4_k_m.gguf", 50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("supervisor.NewWithOpts: %v", err)
	}
	defer sup.Close()

	ttl, err := sup.GetModelTTL("llama-3-8b")
	if err != nil {
		t.Fatalf("GetModelTTL: %v", err)
	}
	if ttl != 50*time.Millisecond {
		t.Errorf("GetModelTTL = %v, want 50ms", ttl)
	}

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	resp.Body.Close()

	insts := fakeHost.Instances()
	if len(insts) == 0 {
		t.Fatalf("expected 1 instance launched")
	}
	inst := insts[len(insts)-1]

	select {
	case <-inst.Done():
		// OK, stopped after per-model 50ms TTL
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected instance to stop after 50ms per-model TTL, but it did not stop within 500ms")
	}

	if err := sup.SetModelTTL("llama-3-8b", 200*time.Millisecond); err != nil {
		t.Fatalf("SetModelTTL: %v", err)
	}
	ttl, _ = sup.GetModelTTL("llama-3-8b")
	if ttl != 200*time.Millisecond {
		t.Errorf("after SetModelTTL, GetModelTTL = %v, want 200ms", ttl)
	}
}

func TestTTL_ExpiryDrainsInFlightSlotWithoutKillingMidGeneration(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

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

	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithDefaultTTL(30*time.Millisecond))
	if err != nil {
		t.Fatalf("supervisor.NewWithOpts: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	respCh := make(chan *http.Response, 1)
	errCh := make(chan error, 1)

	go func() {
		body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			errCh <- err
			return
		}
		respCh <- resp
	}()

	<-slowHandlerStarted

	insts := fakeHost.Instances()
	if len(insts) == 0 {
		t.Fatalf("expected 1 instance launched")
	}
	inst := insts[len(insts)-1]

	time.Sleep(60 * time.Millisecond)

	select {
	case <-inst.Done():
		t.Fatal("instance was killed mid-generation before slot drained!")
	default:
	}

	close(slowHandlerRelease)

	select {
	case err := <-errCh:
		t.Fatalf("request failed: %v", err)
	case resp := <-respCh:
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		resp.Body.Close()
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for response")
	}

	select {
	case <-inst.Done():
		// OK, stopped after slot drained
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected instance to stop after in-flight slot completed and drained")
	}
}

func TestTTL_ZeroTTLNeverExpires(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithDefaultTTL(0))
	if err != nil {
		t.Fatalf("supervisor.NewWithOpts: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	body := `{"model":"llama-3-8b","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	resp.Body.Close()

	insts := fakeHost.Instances()
	if len(insts) == 0 {
		t.Fatalf("expected 1 instance launched")
	}
	inst := insts[len(insts)-1]

	time.Sleep(100 * time.Millisecond)

	select {
	case <-inst.Done():
		t.Fatal("instance stopped when TTL was set to 0 (infinite)")
	default:
		// OK, still running
	}
}
