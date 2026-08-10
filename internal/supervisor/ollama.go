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
	"sort"
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

// ollamaAPIVersion is the version string returned to Ollama clients during handshake.
const ollamaAPIVersion = "0.5.0"

type ollamaVersionResponse struct {
	Version string `json:"version"`
}

type ollamaShowRequest struct {
	Model string `json:"model"`
	Name  string `json:"name"`
}

type ollamaShowResponse struct {
	Modelfile  string             `json:"modelfile"`
	Parameters string             `json:"parameters"`
	Template   string             `json:"template"`
	Details    ollamaModelDetails `json:"details"`
	ModelInfo  map[string]any     `json:"model_info"`
}
type ollamaEmbedRequest struct {
	Model     string          `json:"model"`
	Input     any             `json:"input"`
	Prompt    any             `json:"prompt"`
	KeepAlive json.RawMessage `json:"keep_alive"`
}

type ollamaPSModel struct {
	Name          string             `json:"name"`
	Model         string             `json:"model"`
	Size          int64              `json:"size"`
	Digest        string             `json:"digest"`
	Details       ollamaModelDetails `json:"details"`
	ExpiresAt     time.Time          `json:"expires_at"`
	SizeVRAM      int64              `json:"size_vram"`
	ContextLength uint64             `json:"context_length"`
	ActiveSlots   int                `json:"active_slots"`
	MaxSlots      int                `json:"max_slots"`
}

type ollamaPSResponse struct {
	Models []ollamaPSModel `json:"models"`
}

func (s *Supervisor) handleAPIVersion(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, ollamaVersionResponse{Version: ollamaAPIVersion})
}

func (s *Supervisor) handleAPIShow(w http.ResponseWriter, r *http.Request) {
	var req ollamaShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	ref := req.Model
	if ref == "" {
		ref = req.Name
	}
	if ref == "" {
		writeOllamaError(w, http.StatusBadRequest, "model name is required")
		return
	}

	m, err := s.resolveModel(ref)
	if err != nil {
		var notFound *ModelNotFoundError
		if errors.As(err, &notFound) {
			writeOllamaError(w, http.StatusNotFound, err.Error())
			return
		}
		var ambig *AmbiguousModelError
		if errors.As(err, &ambig) {
			writeOllamaError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeOllamaError(w, http.StatusInternalServerError, err.Error())
		return
	}

	modelInfo := map[string]any{
		"general.architecture": m.Architecture,
	}
	if m.Architecture != "" {
		modelInfo[fmt.Sprintf("%s.context_length", m.Architecture)] = m.ContextLength
		modelInfo[fmt.Sprintf("%s.block_count", m.Architecture)] = m.BlockCount
	} else {
		modelInfo["general.context_length"] = m.ContextLength
	}
	parameters := ""
	if m.ContextLength > 0 {
		parameters = fmt.Sprintf("num_ctx %d", m.ContextLength)
	}

	res := ollamaShowResponse{
		Modelfile:  fmt.Sprintf("FROM %s", m.ID),
		Parameters: parameters,
		Template:   "",
		Details:    modelDetails(m),
		ModelInfo:  modelInfo,
	}

	writeJSON(w, http.StatusOK, res)
}

func (s *Supervisor) handleAPIPs(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	models := make([]ollamaPSModel, 0, len(s.instances))

	for id, ri := range s.instances {
		m, ok := s.models[id]
		if !ok {
			m = Model{
				ID:   id,
				Name: id,
			}
		}
		active, max := ri.Occupancy()
		expiresAt := ri.ExpiresAt()

		vram := m.Size
		alloc := ri.inst.ObservedAllocation()
		if alloc.VRAM > 0 || alloc.RAM > 0 {
			vram = alloc.VRAM
		}

		ctxLen := m.ContextLength
		if te, ok := s.tuned[id]; ok && te.ResultingCtx > 0 {
			ctxLen = te.ResultingCtx
		}

		models = append(models, ollamaPSModel{
			Name:          m.ID,
			Model:         m.ID,
			Size:          m.Size,
			Digest:        m.Digest,
			Details:       modelDetails(m),
			ExpiresAt:     expiresAt,
			SizeVRAM:      vram,
			ContextLength: ctxLen,
			ActiveSlots:   active,
			MaxSlots:      max,
		})
	}
	s.mu.RUnlock()

	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	writeJSON(w, http.StatusOK, ollamaPSResponse{Models: models})
}

type openAIEmbeddingData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type openAIEmbeddingsResponse struct {
	Data []openAIEmbeddingData `json:"data"`
}

func (s *Supervisor) handleAPIEmbed(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req ollamaEmbedRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Model == "" {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	model, err := s.resolveModel(req.Model)
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

	ttl, keepAliveSet := parseKeepAlive(req.KeepAlive)
	if keepAliveSet && ttl != 0 {
		_ = s.SetModelTTL(model.ID, ttl)
	}

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

	if keepAliveSet && ttl == 0 {
		s.mu.Lock()
		resInst := s.instances[model.ID]
		s.mu.Unlock()
		if resInst != nil {
			defer func() {
				release()
				go resInst.drainAndStop(context.Background(), s, "keep_alive=0")
			}()
		} else {
			defer release()
		}
	} else {
		defer release()
	}

	input := req.Input
	if input == nil {
		input = req.Prompt
	}

	openAIReqBody, err := json.Marshal(map[string]any{
		"model": model.ID,
		"input": input,
	})
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "failed to serialize request")
		return
	}

	childURL := inst.URL().String() + "/v1/embeddings"
	childReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, childURL, bytes.NewReader(openAIReqBody))
	if err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "failed to create child request")
		return
	}
	childReq.Header.Set("Content-Type", "application/json")

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

	var openAIResp openAIEmbeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		writeOllamaError(w, http.StatusInternalServerError, "failed to decode child response")
		return
	}

	if r.URL.Path == "/api/embeddings" {
		var emb []float64
		if len(openAIResp.Data) > 0 {
			emb = openAIResp.Data[0].Embedding
		} else {
			emb = []float64{}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"embedding": emb,
		})
		return
	}

	embeddings := make([][]float64, len(openAIResp.Data))
	for i, d := range openAIResp.Data {
		embeddings[i] = d.Embedding
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":      req.Model,
		"embeddings": embeddings,
	})
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

	if ttl, ok := parseKeepAlive(rawKeepAlive); ok && ttl != 0 {
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

	if ttl, ok := parseKeepAlive(rawKeepAlive); ok && ttl == 0 {
		s.mu.Lock()
		resInst := s.instances[model.ID]
		s.mu.Unlock()
		if resInst != nil {
			defer func() {
				go resInst.drainAndStop(context.Background(), s, "keep_alive=0")
			}()
		}
	}
	openAIMessages := make([]map[string]string, len(messages))
	for i, m := range messages {
		openAIMessages[i] = map[string]string{
			"role":    m.Role,
			"content": m.Content,
		}
	}

	openAIReq := map[string]any{
		"model":          model.ID,
		"messages":       openAIMessages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if len(rawOptions) > 0 {
		var opts map[string]any
		if err := json.Unmarshal(rawOptions, &opts); err == nil {
			for k, v := range opts {
				switch k {
				case "model", "messages", "stream", "stream_options":
					// ignore option keys that shadow core parameters
				case "num_predict":
					openAIReq["max_tokens"] = v
				default:
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
	const maxTokenSize = 10 * 1024 * 1024
	scanner.Buffer(make([]byte, 64*1024), maxTokenSize)
	var fullContent strings.Builder
	var finishReason string
	var evalCount int
	var promptEvalCount int
	var firstTokenReceived bool
	var promptEvalDuration time.Duration
	var evalDuration time.Duration
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
			if !firstTokenReceived {
				firstTokenReceived = true
				promptEvalDuration = time.Since(startTime)
				evalStart = time.Now()
			}
			if chunk.Usage == nil {
				evalCount++
			}
			if stream {
				enc := json.NewEncoder(w)
				if isChat {
					_ = enc.Encode(ollamaChatIntermediateChunk{
						Model:     modelRef,
						CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
						Message:   ollamaMessage{Role: "assistant", Content: contentChunk},
						Done:      false,
					})
				} else {
					_ = enc.Encode(ollamaGenerateIntermediateChunk{
						Model:     modelRef,
						CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
						Response:  contentChunk,
						Done:      false,
					})
				}
				flusher.Flush()
			} else {
				fullContent.WriteString(contentChunk)
			}
		}
	}

	if err := scanner.Err(); err != nil && !stream {
		writeOllamaError(w, http.StatusInternalServerError, fmt.Sprintf("stream scanner error: %v", err))
		return
	}

	totalDuration := time.Since(startTime)
	if firstTokenReceived {
		evalDuration = time.Since(evalStart)
	} else {
		promptEvalDuration = totalDuration
		evalDuration = 0
	}
	if finishReason == "" {
		finishReason = "stop"
	}

	if stream {
		enc := json.NewEncoder(w)
		if isChat {
			_ = enc.Encode(ollamaChatFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Message:            ollamaMessage{Role: "assistant", Content: ""},
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: promptEvalDuration.Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		} else {
			_ = enc.Encode(ollamaGenerateFinalChunk{
				Model:              modelRef,
				CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
				Response:           "",
				Done:               true,
				DoneReason:         finishReason,
				TotalDuration:      totalDuration.Nanoseconds(),
				LoadDuration:       loadDuration.Nanoseconds(),
				PromptEvalCount:    promptEvalCount,
				PromptEvalDuration: promptEvalDuration.Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		}
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
				PromptEvalDuration: promptEvalDuration.Nanoseconds(),
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
				PromptEvalDuration: promptEvalDuration.Nanoseconds(),
				EvalCount:          evalCount,
				EvalDuration:       evalDuration.Nanoseconds(),
			})
		}
	}
}
