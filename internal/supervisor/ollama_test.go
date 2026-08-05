package supervisor_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

func TestOllamaChat_StreamingAndTerminalStats(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	reqBody := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hello"}],"stream":true}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "application/x-ndjson") {
		t.Errorf("Content-Type = %q, want application/x-ndjson", contentType)
	}

	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner error: %v", err)
	}

	if len(lines) < 2 {
		t.Fatalf("got %d NDJSON lines, want at least 2", len(lines))
	}

	// Verify intermediate chunks
	var fullText string
	for i := range len(lines) - 1 {
		var chunk struct {
			Model     string `json:"model"`
			CreatedAt string `json:"created_at"`
			Message   struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal([]byte(lines[i]), &chunk); err != nil {
			t.Fatalf("line %d unmarshal: %v", i, err)
		}
		if chunk.Done {
			t.Errorf("intermediate chunk %d has done=true", i)
		}
		if chunk.Message.Role != "assistant" {
			t.Errorf("intermediate chunk %d role = %q, want assistant", i, chunk.Message.Role)
		}
		fullText += chunk.Message.Content
	}

	if fullText == "" {
		t.Error("expected non-empty concatenated message content")
	}

	// Verify terminal statistics object
	lastLine := lines[len(lines)-1]
	var finalObj struct {
		Model     string `json:"model"`
		CreatedAt string `json:"created_at"`
		Message   struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Done               bool   `json:"done"`
		DoneReason         string `json:"done_reason"`
		TotalDuration      int64  `json:"total_duration"`
		LoadDuration       int64  `json:"load_duration"`
		PromptEvalCount    int    `json:"prompt_eval_count"`
		PromptEvalDuration int64  `json:"prompt_eval_duration"`
		EvalCount          int    `json:"eval_count"`
		EvalDuration       int64  `json:"eval_duration"`
	}
	if err := json.Unmarshal([]byte(lastLine), &finalObj); err != nil {
		t.Fatalf("terminal obj unmarshal: %v", err)
	}

	if !finalObj.Done {
		t.Error("terminal object done = false, want true")
	}
	if finalObj.DoneReason != "stop" {
		t.Errorf("terminal object done_reason = %q, want 'stop'", finalObj.DoneReason)
	}
	if finalObj.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", finalObj.TotalDuration)
	}
	if finalObj.LoadDuration < 0 {
		t.Errorf("load_duration = %d, want >= 0", finalObj.LoadDuration)
	}
	if finalObj.EvalCount <= 0 {
		t.Errorf("eval_count = %d, want > 0", finalObj.EvalCount)
	}
}

func TestOllamaChat_NonStreaming(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	reqBody := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hello"}],"stream":false}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var res struct {
		Model     string `json:"model"`
		CreatedAt string `json:"created_at"`
		Message   struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Done               bool   `json:"done"`
		DoneReason         string `json:"done_reason"`
		TotalDuration      int64  `json:"total_duration"`
		LoadDuration       int64  `json:"load_duration"`
		PromptEvalCount    int    `json:"prompt_eval_count"`
		PromptEvalDuration int64  `json:"prompt_eval_duration"`
		EvalCount          int    `json:"eval_count"`
		EvalDuration       int64  `json:"eval_duration"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if !res.Done {
		t.Error("done = false, want true")
	}
	if res.Message.Role != "assistant" {
		t.Errorf("message.role = %q, want assistant", res.Message.Role)
	}
	if res.Message.Content == "" {
		t.Error("message.content is empty")
	}
	if res.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", res.TotalDuration)
	}
}

func TestOllamaGenerate_StreamingAndTerminalStats(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	reqBody := `{"model":"llama-3-8b:q4_k_m","prompt":"Why is the sky blue?","system":"Be concise","stream":true}`
	resp, err := http.Post(srv.URL+"/api/generate", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/generate: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if len(lines) < 2 {
		t.Fatalf("got %d lines, want at least 2", len(lines))
	}

	// Verify intermediate chunks use "response" field
	for i := range len(lines) - 1 {
		var chunk struct {
			Model    string `json:"model"`
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := json.Unmarshal([]byte(lines[i]), &chunk); err != nil {
			t.Fatalf("line %d unmarshal: %v", i, err)
		}
		if chunk.Done {
			t.Errorf("intermediate chunk %d has done=true", i)
		}
	}

	// Terminal object
	var finalObj struct {
		Model         string `json:"model"`
		Response      string `json:"response"`
		Done          bool   `json:"done"`
		DoneReason    string `json:"done_reason"`
		TotalDuration int64  `json:"total_duration"`
		LoadDuration  int64  `json:"load_duration"`
		EvalCount     int    `json:"eval_count"`
		EvalDuration  int64  `json:"eval_duration"`
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &finalObj); err != nil {
		t.Fatalf("terminal obj unmarshal: %v", err)
	}
	if !finalObj.Done {
		t.Error("done = false, want true")
	}
	if finalObj.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", finalObj.TotalDuration)
	}
}

func TestOllamaGenerate_NonStreaming(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	reqBody := `{"model":"llama-3-8b:q4_k_m","prompt":"Hello","stream":false}`
	resp, err := http.Post(srv.URL+"/api/generate", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/generate: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var res struct {
		Model         string `json:"model"`
		Response      string `json:"response"`
		Done          bool   `json:"done"`
		DoneReason    string `json:"done_reason"`
		TotalDuration int64  `json:"total_duration"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !res.Done {
		t.Error("done = false, want true")
	}
	if res.Response == "" {
		t.Error("response content is empty")
	}
	if res.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", res.TotalDuration)
	}
}

func TestOllama_KeepAliveSetsModelTTL(t *testing.T) {
	fakeHost := host.NewFakeHost()
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	reqBody := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hi"}],"keep_alive":"150ms"}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	ttl, err := sup.GetModelTTL("llama-3-8b:q4_k_m")
	if err != nil {
		t.Fatalf("GetModelTTL: %v", err)
	}

	if ttl != 150*time.Millisecond {
		t.Errorf("ttl = %v, want 150ms", ttl)
	}
}

func TestOllama_UnknownModelError(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	reqBody := `{"model":"nonexistent-model","messages":[{"role":"user","content":"Hi"}]}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}

	var errResp struct {
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if !strings.Contains(errResp.Error, "model 'nonexistent-model' not found") {
		t.Errorf("error = %q, want to contain model not found error", errResp.Error)
	}
}

func TestOllama_WireFormatFixtures(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	// 1. Chat streaming framing & terminal statistics schema assertion
	chatReq := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(chatReq))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("chat stream returned %d lines, want >= 2", len(lines))
	}

	// Verify intermediate NDJSON chunk structure
	var interChunk map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &interChunk); err != nil {
		t.Fatalf("unmarshal intermediate chat chunk: %v", err)
	}
	for _, forbidden := range []string{"total_duration", "done_reason", "eval_count"} {
		if _, ok := interChunk[forbidden]; ok {
			t.Errorf("intermediate chunk unexpected key %q", forbidden)
		}
	}

	// Read recorded terminal fixture
	fixtureData, err := os.ReadFile("testdata/ollama_chat_stream.ndjson")
	if err != nil {
		t.Fatalf("read testdata/ollama_chat_stream.ndjson: %v", err)
	}
	fixtureLines := strings.Split(strings.TrimSpace(string(fixtureData)), "\n")
	var expectedTerminal map[string]any
	_ = json.Unmarshal([]byte(fixtureLines[len(fixtureLines)-1]), &expectedTerminal)

	var actualTerminal map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &actualTerminal); err != nil {
		t.Fatalf("unmarshal actual terminal chunk: %v", err)
	}

	for key := range expectedTerminal {
		if _, ok := actualTerminal[key]; !ok {
			t.Errorf("terminal chat response missing key %q present in recorded fixture", key)
		}
	}

	// 2. Generate streaming wire format
	genReq := `{"model":"llama-3-8b:q4_k_m","prompt":"Hello","stream":true}`
	respGen, err := http.Post(srv.URL+"/api/generate", "application/json", strings.NewReader(genReq))
	if err != nil {
		t.Fatalf("POST /api/generate: %v", err)
	}
	defer respGen.Body.Close()

	bufGen := new(bytes.Buffer)
	_, _ = bufGen.ReadFrom(respGen.Body)
	genLines := strings.Split(strings.TrimSpace(bufGen.String()), "\n")
	if len(genLines) < 2 {
		t.Fatalf("generate stream returned %d lines, want >= 2", len(genLines))
	}

	var interGen map[string]any
	if err := json.Unmarshal([]byte(genLines[0]), &interGen); err != nil {
		t.Fatalf("unmarshal intermediate generate chunk: %v", err)
	}
	if _, ok := interGen["response"]; !ok {
		t.Error("intermediate generate chunk missing 'response' key")
	}
}

func TestOllama_KeepAliveZeroUnloadsImmediately(t *testing.T) {
	fakeHost := host.NewFakeHost()
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	sup, err := supervisor.New(fakeHost, tmpDir)
	if err != nil {
		t.Fatalf("supervisor.New: %v", err)
	}
	defer sup.Close()

	srv := httptest.NewServer(sup.Handler())
	defer srv.Close()

	// Send keep_alive: 0
	reqBody := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hi"}],"keep_alive":0}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	// Wait briefly for drain & unload goroutine to complete
	insts := fakeHost.Instances()
	if len(insts) > 0 {
		select {
		case <-insts[0].Done():
			// OK, instance unloaded immediately after request
		case <-time.After(500 * time.Millisecond):
			t.Fatal("expected instance to unload immediately after keep_alive=0 request, but it remained running")
		}
	}
}

func TestOllama_ParseKeepAliveVariants(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		wantDur time.Duration
		wantOk  bool
	}{
		{"duration string 5m", `"5m"`, 5 * time.Minute, true},
		{"duration string 10s", `"10s"`, 10 * time.Second, true},
		{"numeric string 300", `"300"`, 300 * time.Second, true},
		{"string 0", `"0"`, 0, true},
		{"string 0s", `"0s"`, 0, true},
		{"string -1", `"-1"`, -1, true},
		{"string -1s", `"-1s"`, -1, true},
		{"number 300", `300`, 300 * time.Second, true},
		{"number 0", `0`, 0, true},
		{"number -1", `-1`, -1, true},
		{"null", `null`, 0, false},
		{"empty", ``, 0, false},
		{"invalid string", `"invalid"`, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDur, gotOk := supervisor.ExportParseKeepAlive(json.RawMessage(tt.raw))
			if gotOk != tt.wantOk || gotDur != tt.wantDur {
				t.Errorf("parseKeepAlive(%s) = (%v, %v), want (%v, %v)", tt.raw, gotDur, gotOk, tt.wantDur, tt.wantOk)
			}
		})
	}
}
func TestOllama_APIVersion(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/version")
	if err != nil {
		t.Fatalf("GET /api/version: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var res struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if res.Version == "" {
		t.Errorf("version is empty")
	}
}

func TestOllama_APIShow(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	srv, _ := newTestServer(t, tmpDir)
	defer srv.Close()

	// 1. Success case using model key
	body := `{"model":"llama-3-8b:q4_k_m"}`
	resp, err := http.Post(srv.URL+"/api/show", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var showRes struct {
		Modelfile  string `json:"modelfile"`
		Parameters string `json:"parameters"`
		Template   string `json:"template"`
		Details    struct {
			Format            string `json:"format"`
			Family            string `json:"family"`
			QuantizationLevel string `json:"quantization_level"`
		} `json:"details"`
		ModelInfo map[string]any `json:"model_info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&showRes); err != nil {
		t.Fatalf("failed to decode show response: %v", err)
	}

	if showRes.Details.Format != "gguf" {
		t.Errorf("details.format = %q, want gguf", showRes.Details.Format)
	}
	if showRes.Details.Family != "llama" {
		t.Errorf("details.family = %q, want llama", showRes.Details.Family)
	}
	if showRes.Details.QuantizationLevel != "Q4_K_M" {
		t.Errorf("details.quantization_level = %q, want Q4_K_M", showRes.Details.QuantizationLevel)
	}

	if arch, ok := showRes.ModelInfo["general.architecture"].(string); !ok || arch != "llama" {
		t.Errorf("model_info[general.architecture] = %v, want llama", showRes.ModelInfo["general.architecture"])
	}
	if _, ok := showRes.ModelInfo["llama.context_length"]; !ok {
		t.Errorf("model_info[llama.context_length] missing")
	}

	if !strings.Contains(showRes.Modelfile, "llama-3-8b:q4_k_m") {
		t.Errorf("modelfile = %q, expected to contain model ID", showRes.Modelfile)
	}

	// 2. Success case using name key
	bodyName := `{"name":"llama-3-8b:q4_k_m"}`
	respName, err := http.Post(srv.URL+"/api/show", "application/json", strings.NewReader(bodyName))
	if err != nil {
		t.Fatalf("POST /api/show (name): %v", err)
	}
	respName.Body.Close()
	if respName.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 for name parameter", respName.StatusCode)
	}

	// 3. Not found error
	bodyNotFound := `{"model":"nonexistent:tag"}`
	respNotFound, err := http.Post(srv.URL+"/api/show", "application/json", strings.NewReader(bodyNotFound))
	if err != nil {
		t.Fatalf("POST /api/show not found: %v", err)
	}
	respNotFound.Body.Close()
	if respNotFound.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404 for unknown model", respNotFound.StatusCode)
	}

	// 4. Bad request error (empty body / no model specified)
	bodyEmpty := `{}`
	respEmpty, err := http.Post(srv.URL+"/api/show", "application/json", strings.NewReader(bodyEmpty))
	if err != nil {
		t.Fatalf("POST /api/show empty: %v", err)
	}
	respEmpty.Body.Close()
	if respEmpty.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for empty request", respEmpty.StatusCode)
	}
}

func TestOllama_APIPs(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	srv, _ := newTestServer(t, tmpDir)
	defer srv.Close()

	// 1. Initial GET /api/ps -> no resident instances
	resp, err := http.Get(srv.URL + "/api/ps")
	if err != nil {
		t.Fatalf("GET /api/ps: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var psRes struct {
		Models []struct {
			Name        string `json:"name"`
			Model       string `json:"model"`
			Size        int64  `json:"size"`
			SizeVRAM    int64  `json:"size_vram"`
			ActiveSlots int    `json:"active_slots"`
			MaxSlots    int    `json:"max_slots"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&psRes); err != nil {
		t.Fatalf("failed to decode ps response: %v", err)
	}
	if len(psRes.Models) != 0 {
		t.Errorf("got %d models before load, want 0", len(psRes.Models))
	}

	// 2. Trigger non-streaming chat request to make llama-3-8b:q4_k_m resident
	chatBody := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hi"}],"stream":false}`
	chatResp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(chatBody))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	chatResp.Body.Close()

	// 3. GET /api/ps -> should report resident model and slot info
	respResident, err := http.Get(srv.URL + "/api/ps")
	if err != nil {
		t.Fatalf("GET /api/ps after load: %v", err)
	}
	defer respResident.Body.Close()

	if respResident.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", respResident.StatusCode)
	}

	if err := json.NewDecoder(respResident.Body).Decode(&psRes); err != nil {
		t.Fatalf("failed to decode ps response after load: %v", err)
	}
	if len(psRes.Models) != 1 {
		t.Fatalf("got %d models after load, want 1", len(psRes.Models))
	}

	m := psRes.Models[0]
	if m.Name != "llama-3-8b:q4_k_m" {
		t.Errorf("model.name = %q, want llama-3-8b:q4_k_m", m.Name)
	}
	if m.Size <= 0 {
		t.Errorf("model.size = %d, want > 0", m.Size)
	}
	if m.SizeVRAM <= 0 {
		t.Errorf("model.size_vram = %d, want > 0", m.SizeVRAM)
	}
	if m.MaxSlots < 1 {
		t.Errorf("model.max_slots = %d, want >= 1", m.MaxSlots)
	}
}
func TestOllama_APIEmbed(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "llama-3-8b.q4_k_m.gguf", "llama", "Q4_K_M")

	srv, fakeHost := newTestServer(t, tmpDir)
	defer srv.Close()

	// 1. Success POST /api/embed
	body := `{"model":"llama-3-8b:q4_k_m","input":"Hello world"}`
	resp, err := http.Post(srv.URL+"/api/embed", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /api/embed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var resEmbed struct {
		Model      string      `json:"model"`
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&resEmbed); err != nil {
		t.Fatalf("decoding /api/embed response: %v", err)
	}
	if len(resEmbed.Embeddings) != 1 || len(resEmbed.Embeddings[0]) != 3 {
		t.Errorf("got embeddings = %v, want 1 vector of length 3", resEmbed.Embeddings)
	}
	if resEmbed.Model != "llama-3-8b:q4_k_m" {
		t.Errorf("got model = %q, want llama-3-8b:q4_k_m", resEmbed.Model)
	}

	// Verify instance is resident in FakeHost
	if len(fakeHost.Instances()) != 1 {
		t.Errorf("len(fakeHost.Instances()) = %d, want 1", len(fakeHost.Instances()))
	}

	// 2. Success POST /api/embeddings (legacy endpoint)
	bodyLegacy := `{"model":"llama-3-8b:q4_k_m","prompt":"Hello world"}`
	respLegacy, err := http.Post(srv.URL+"/api/embeddings", "application/json", strings.NewReader(bodyLegacy))
	if err != nil {
		t.Fatalf("POST /api/embeddings: %v", err)
	}
	defer respLegacy.Body.Close()

	if respLegacy.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", respLegacy.StatusCode)
	}
	var resLegacy struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(respLegacy.Body).Decode(&resLegacy); err != nil {
		t.Fatalf("decoding /api/embeddings response: %v", err)
	}
	if len(resLegacy.Embedding) != 3 {
		t.Errorf("got embedding = %v, want vector of length 3", resLegacy.Embedding)
	}
	// 3. Unknown model: 404 Ollama error format
	unknownBody := `{"model":"unknown-model:q4_k_m","input":"Hi"}`
	resp404, err := http.Post(srv.URL+"/api/embed", "application/json", strings.NewReader(unknownBody))
	if err != nil {
		t.Fatalf("POST /api/embed 404: %v", err)
	}
	defer resp404.Body.Close()

	if resp404.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp404.StatusCode)
	}
	var errRes struct {
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp404.Body).Decode(&errRes); err != nil {
		t.Fatalf("decoding error response: %v", err)
	}
	if errRes.Error != "model 'unknown-model:q4_k_m' not found" {
		t.Errorf("error string = %q, want model 'unknown-model:q4_k_m' not found", errRes.Error)
	}
}
