package supervisor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	Options   json.RawMessage `json:"options,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

type ollamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	System    string          `json:"system,omitempty"`
	Template  string          `json:"template,omitempty"`
	Context   []int           `json:"context,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Options   json.RawMessage `json:"options,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

type ollamaChatIntermediateChunk struct {
	Model     string        `json:"model"`
	CreatedAt string        `json:"created_at"`
	Message   ollamaMessage `json:"message"`
	Done      bool          `json:"done"`
}

type ollamaChatFinalChunk struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	TotalDuration      int64         `json:"total_duration"`
	LoadDuration       int64         `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int64         `json:"eval_duration"`
}

type ollamaGenerateIntermediateChunk struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
	Done      bool   `json:"done"`
}

type ollamaGenerateFinalChunk struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

type openAIChunkChoiceDelta struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChunkChoice struct {
	Delta        openAIChunkChoiceDelta `json:"delta"`
	FinishReason *string                `json:"finish_reason"`
}

type openAIChunkUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type openAIChunk struct {
	Choices []openAIChunkChoice `json:"choices"`
	Usage   *openAIChunkUsage   `json:"usage"`
}

func writeOllamaError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func parseKeepAlive(raw json.RawMessage) (time.Duration, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		s = strings.TrimSpace(s)
		if s == "" {
			return 0, false
		}
		if s == "-1" || s == "-1s" {
			return -1, true
		}
		if s == "0" || s == "0s" {
			return 0, true
		}
		d, err := time.ParseDuration(s)
		if err == nil {
			return d, true
		}
		if f, err := strconv.ParseFloat(s, 64); err == nil {
			if f < 0 {
				return -1, true
			}
			return time.Duration(f * float64(time.Second)), true
		}
		return 0, false
	}

	var n float64
	if err := json.Unmarshal(raw, &n); err == nil {
		if n < 0 {
			return -1, true
		}
		return time.Duration(n * float64(time.Second)), true
	}

	return 0, false
}

func (s *Supervisor) handleAPIChat(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req ollamaChatRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Model == "" {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	s.serveOllamaCompletion(w, r, req.Model, req.Messages, req.Options, req.KeepAlive, stream, true)
}

func (s *Supervisor) handleAPIGenerate(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req ollamaGenerateRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Model == "" {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	messages := make([]ollamaMessage, 0, 2)
	if req.System != "" {
		messages = append(messages, ollamaMessage{Role: "system", Content: req.System})
	}
	messages = append(messages, ollamaMessage{Role: "user", Content: req.Prompt})

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	s.serveOllamaCompletion(w, r, req.Model, messages, req.Options, req.KeepAlive, stream, false)
}

func (s *Supervisor) serveOllamaCompletion(
	w http.ResponseWriter,
	r *http.Request,
	modelRef string,
	messages []ollamaMessage,
	rawOptions json.RawMessage,
	rawKeepAlive json.RawMessage,
	stream bool,
	isChat bool,
) {
	model, err := s.resolveModel(modelRef)
	if err != nil {
		var notFound *ModelNotFoundError
		var ambiguous *AmbiguousModelError
		switch {
		case errors.As(err, &notFound):
			writeOllamaError(w, http.StatusNotFound, fmt.Sprintf("model '%s' not found", notFound.Ref))
		case errors.As(err, &ambiguous):
			msg := fmt.Sprintf("model '%s' is ambiguous; available tags: %s", ambiguous.Name, strings.Join(ambiguous.Tags, ", "))
			writeOllamaError(w, http.StatusBadRequest, msg)
		default:
			writeOllamaError(w, http.StatusInternalServerError, err.Error())
		}
		return
	}

	if ttl, ok := parseKeepAlive(rawKeepAlive); ok {
		_ = s.SetModelTTL(model.ID, ttl)
	}

	loadStart := time.Now()
	inst, release, err := s.getOrLaunchInstance(r.Context(), model)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		status := http.StatusInternalServerError
		if err.Error() == "supervisor is closed" {
			status = http.StatusServiceUnavailable
		}
		writeOllamaError(w, status, err.Error())
		return
	}
	defer release()
	loadDuration := time.Since(loadStart)

	openAIMessages := make([]map[string]string, len(messages))
	for i, m := range messages {
		openAIMessages[i] = map[string]string{
			"role":    m.Role,
			"content": m.Content,
		}
	}

	openAIReq := map[string]any{
		"model":    model.ID,
		"messages": openAIMessages,
		"stream":   true,
	}

	if len(rawOptions) > 0 {
		var opts map[string]any
		if err := json.Unmarshal(rawOptions, &opts); err == nil {
			for k, v := range opts {
				if k == "num_predict" {
					openAIReq["max_tokens"] = v
				} else {
					openAIReq[k] = v
				}
			}
		}
	}

	reqBytes, err := json.Marshal(openAIReq)
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "failed to serialize request")
		return
	}

	childURL := inst.URL().String() + "/v1/chat/completions"
	childReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, childURL, bytes.NewReader(reqBytes))
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "failed to create child request")
		return
	}
	childReq.Header.Set("Content-Type", "application/json")

	startTime := time.Now()
	resp, err := http.DefaultClient.Do(childReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		writeOllamaError(w, http.StatusServiceUnavailable, fmt.Sprintf("proxy error: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		writeOllamaError(w, resp.StatusCode, string(respBody))
		return
	}

	var flusher http.Flusher
	if stream {
		f, ok := w.(http.Flusher)
		if !ok {
			writeOllamaError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}
		flusher = f
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
	}

	scanner := bufio.NewScanner(resp.Body)
	var fullContent strings.Builder
	var finishReason string
	var evalCount int
	var promptEvalCount int
	evalStart := time.Now()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "[DONE]" {
			break
		}

		var chunk openAIChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				promptEvalCount = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				evalCount = chunk.Usage.CompletionTokens
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			finishReason = *choice.FinishReason
		}

		contentChunk := choice.Delta.Content
		if contentChunk != "" {
			if chunk.Usage == nil {
				evalCount++
			}
			if stream {
				var lineBytes []byte
				if isChat {
					lineBytes, _ = json.Marshal(ollamaChatIntermediateChunk{
						Model:     modelRef,
						CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
						Message:   ollamaMessage{Role: "assistant", Content: contentChunk},
						Done:      false,
					})
				} else {
					lineBytes, _ = json.Marshal(ollamaGenerateIntermediateChunk{
						Model:     modelRef,
						CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
						Response:  contentChunk,
						Done:      false,
					})
				}
				_, _ = w.Write(append(lineBytes, '\n'))
				flusher.Flush()
			} else {
				fullContent.WriteString(contentChunk)
			}
		}
	}

	totalDuration := time.Since(startTime)
	evalDuration := time.Since(evalStart)
	if finishReason == "" {
		finishReason = "stop"
	}

	if stream {
		var finalBytes []byte
		if isChat {
			finalBytes, _ = json.Marshal(ollamaChatFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Message:            ollamaMessage{Role: "assistant", Content: ""},
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: (1 * time.Millisecond).Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		} else {
			finalBytes, _ = json.Marshal(ollamaGenerateFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Response:           "",
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: (1 * time.Millisecond).Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		}
		_, _ = w.Write(append(finalBytes, '\n'))
		flusher.Flush()
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if isChat {
			_ = json.NewEncoder(w).Encode(ollamaChatFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Message:            ollamaMessage{Role: "assistant", Content: fullContent.String()},
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: (1 * time.Millisecond).Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		} else {
			_ = json.NewEncoder(w).Encode(ollamaGenerateFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Response:           fullContent.String(),
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: (1 * time.Millisecond).Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		}
	}
}
