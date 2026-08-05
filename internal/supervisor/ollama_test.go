package supervisor_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

	// Assert Chat Streaming Wire Format against expected JSON schema keys
	chatReq := `{"model":"llama-3-8b:q4_k_m","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(chatReq))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")

	if len(lines) == 0 {
		t.Fatal("empty response")
	}

	// Verify terminal chunk keys match Ollama wire format specification exactly
	lastLine := lines[len(lines)-1]
	var rawKeys map[string]any
	if err := json.Unmarshal([]byte(lastLine), &rawKeys); err != nil {
		t.Fatalf("unmarshal terminal payload: %v", err)
	}

	expectedKeys := []string{
		"model", "created_at", "message", "done", "done_reason",
		"total_duration", "load_duration", "prompt_eval_count",
		"prompt_eval_duration", "eval_count", "eval_duration",
	}

	for _, key := range expectedKeys {
		if _, ok := rawKeys[key]; !ok {
			t.Errorf("missing key %q in terminal Ollama payload fixture: %s", key, lastLine)
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
