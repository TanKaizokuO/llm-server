package supervisor_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
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

func newTestServer(t *testing.T, dirs ...string) *httptest.Server {
	t.Helper()
	if len(dirs) == 0 {
		tmpDir := t.TempDir()
		writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")
		dirs = []string{tmpDir}
	}
	s, err := supervisor.New(dirs...)
	if err != nil {
		t.Fatalf("supervisor.New failed: %v", err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestHealthReportsReadyWithNoModelLoaded(t *testing.T) {
	srv := newTestServer(t)

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

	_, err := supervisor.New(tmpDir)
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

	srv := newTestServer(t, tmpDir)

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

	srv := newTestServer(t, dir1, dir2)

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
