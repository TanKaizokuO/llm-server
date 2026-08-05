// Package supervisor implements the llm-server Supervisor: the daemon that
// discovers Models, supervises llama-server Instances, and proxies the Ollama
// and OpenAI HTTP surfaces onto them.
//
// The Supervisor never performs inference. It links no inference engine and
// executes no model graph; every generation is proxied to a child
// llama-server Instance. That constraint is what keeps this a narrow tool
// rather than a platform, and it is not negotiable.
package supervisor

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
)

type loadRequest struct {
	ready chan struct{}
	inst  host.Instance
	err   error
}

// Supervisor is the daemon's root object. It owns Model discovery, the HTTP surface,
// and, in later work, the Instance registry and the Tuning cache.
type Supervisor struct {
	mu         sync.RWMutex
	host       host.Host
	models     map[string]Model
	modelsList []Model
	instances  map[string]host.Instance
	loading    map[string]*loadRequest
	closed     bool
}

// New builds a Supervisor by scanning the configured directories for Models.
// Host is the single injected boundary in the system representing physical hardware.
// If h is nil, a default real Host is used.
func New(h host.Host, dirs ...string) (*Supervisor, error) {
	if h == nil {
		h = host.New()
	}
	models, err := discoverModels(dirs)
	if err != nil {
		return nil, err
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in scanned directories: %v", dirs)
	}

	modelMap := make(map[string]Model, len(models))
	for _, m := range models {
		modelMap[m.ID] = m
	}

	return &Supervisor{
		host:       h,
		models:     modelMap,
		modelsList: models,
		instances:  make(map[string]host.Instance),
		loading:    make(map[string]*loadRequest),
	}, nil
}

// Handler returns the Supervisor's HTTP router. This is the single router the
// binary serves and the one tests drive; there is no separate test wiring.
func (s *Supervisor) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /api/tags", s.handleAPITags)
	mux.HandleFunc("GET /v1/models", s.handleV1Models)
	mux.HandleFunc("POST /v1/chat/completions", s.handleV1ChatCompletions)
	return mux
}

// Evict stops and removes the resident Instance for the specified model reference if one exists.
// It returns nil if no Instance was resident for the model.
func (s *Supervisor) Evict(ctx context.Context, ref string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	m, err := s.resolveModel(ref)
	modelID := ref
	if err == nil {
		modelID = m.ID
	}

	s.mu.Lock()
	inst, ok := s.instances[modelID]
	if ok {
		delete(s.instances, modelID)
	}
	req, loading := s.loading[modelID]
	s.mu.Unlock()

	var stopErr error
	if ok && inst != nil {
		stopCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		stopErr = inst.Stop(stopCtx)
	}

	if loading && req != nil {
		select {
		case <-req.ready:
			s.mu.Lock()
			loadedInst, loadedOk := s.instances[modelID]
			if loadedOk {
				delete(s.instances, modelID)
			}
			s.mu.Unlock()
			if loadedOk && loadedInst != nil {
				stopCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
				defer cancel()
				if err := loadedInst.Stop(stopCtx); err != nil && stopErr == nil {
					stopErr = err
				}
			}
		case <-ctx.Done():
			if stopErr == nil {
				stopErr = ctx.Err()
			}
		}
	}

	return stopErr
}

// Close stops all resident Instances supervised by the daemon.
func (s *Supervisor) Close() error {
	s.mu.Lock()
	s.closed = true
	instancesToStop := make([]host.Instance, 0, len(s.instances))
	for _, inst := range s.instances {
		instancesToStop = append(instancesToStop, inst)
	}
	s.instances = make(map[string]host.Instance)
	inFlight := make([]*loadRequest, 0, len(s.loading))
	for _, req := range s.loading {
		inFlight = append(inFlight, req)
	}
	s.mu.Unlock()

	stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var errs []error
	for _, inst := range instancesToStop {
		if err := inst.Stop(stopCtx); err != nil {
			errs = append(errs, fmt.Errorf("stopping instance: %w", err))
		}
	}

	for _, req := range inFlight {
		select {
		case <-req.ready:
			if req.inst != nil {
				if err := req.inst.Stop(stopCtx); err != nil {
					errs = append(errs, fmt.Errorf("stopping in-flight instance: %w", err))
				}
			}
		case <-stopCtx.Done():
			errs = append(errs, fmt.Errorf("timed out waiting for in-flight load to stop: %w", stopCtx.Err()))
		}
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

// handleHealth reports that the Supervisor process itself is up and able to
// accept requests. It is deliberately independent of any Model: a process
// supervisor managing llm-server needs to know that the daemon is alive, not
// that some Model happens to be resident.
func (s *Supervisor) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

type ollamaModelDetails struct {
	Format            string `json:"format"`
	Family            string `json:"family"`
	Families          any    `json:"families"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
}

type ollamaModel struct {
	Name       string             `json:"name"`
	Model      string             `json:"model"`
	ModifiedAt time.Time          `json:"modified_at"`
	Size       int64              `json:"size"`
	Digest     string             `json:"digest"`
	Details    ollamaModelDetails `json:"details"`
}

type ollamaTagsResponse struct {
	Models []ollamaModel `json:"models"`
}

func (s *Supervisor) handleAPITags(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	models := s.modelsList
	s.mu.RUnlock()

	res := ollamaTagsResponse{
		Models: make([]ollamaModel, 0, len(models)),
	}

	for _, m := range models {
		res.Models = append(res.Models, ollamaModel{
			Name:       m.ID,
			Model:      m.ID,
			ModifiedAt: m.ModTime,
			Size:       m.Size,
			Digest:     m.Digest,
			Details: ollamaModelDetails{
				Format:            "gguf",
				Family:            m.Architecture,
				Families:          nil,
				ParameterSize:     "",
				QuantizationLevel: m.Quantization,
			},
		})
	}

	writeJSON(w, http.StatusOK, res)
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openAIModelsResponse struct {
	Object string        `json:"object"`
	Data   []openAIModel `json:"data"`
}

func (s *Supervisor) handleV1Models(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	models := s.modelsList
	s.mu.RUnlock()

	res := openAIModelsResponse{
		Object: "list",
		Data:   make([]openAIModel, 0, len(models)),
	}

	for _, m := range models {
		res.Data = append(res.Data, openAIModel{
			ID:      m.ID,
			Object:  "model",
			Created: m.ModTime.Unix(),
			OwnedBy: "llm-server",
		})
	}

	writeJSON(w, http.StatusOK, res)
}

type openAIChatCompletionRequest struct {
	Model string `json:"model"`
}

type openAIErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

type openAIErrorResponse struct {
	Error openAIErrorDetail `json:"error"`
}

func writeOpenAIError(w http.ResponseWriter, status int, message, param, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(openAIErrorResponse{
		Error: openAIErrorDetail{
			Message: message,
			Type:    "invalid_request_error",
			Param:   param,
			Code:    code,
		},
	})
}

// ModelNotFoundError indicates that a requested model reference did not match any discovered model.
type ModelNotFoundError struct {
	Ref string
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("model %q not found", e.Ref)
}

// AmbiguousModelError indicates that a bare model name matches multiple discovered quantisations.
type AmbiguousModelError struct {
	Name string
	Tags []string
}

func (e *AmbiguousModelError) Error() string {
	return fmt.Sprintf("model %q is ambiguous; available tags: %s", e.Name, strings.Join(e.Tags, ", "))
}

func (s *Supervisor) resolveModel(ref string) (Model, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if m, ok := s.models[ref]; ok {
		return m, nil
	}

	var matches []Model
	for _, m := range s.modelsList {
		if m.Name == ref {
			matches = append(matches, m)
		}
	}

	if len(matches) == 0 {
		return Model{}, &ModelNotFoundError{Ref: ref}
	}
	if len(matches) == 1 {
		return matches[0], nil
	}

	tags := make([]string, 0, len(matches))
	for _, m := range matches {
		tags = append(tags, m.Tag)
	}
	return Model{}, &AmbiguousModelError{Name: ref, Tags: tags}
}

func (s *Supervisor) handleV1ChatCompletions(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "failed to read request body", "", "invalid_request")
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req openAIChatCompletionRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Model == "" {
		writeOpenAIError(w, http.StatusBadRequest, "invalid request body", "model", "invalid_request")
		return
	}

	model, err := s.resolveModel(req.Model)
	if err != nil {
		var notFound *ModelNotFoundError
		var ambiguous *AmbiguousModelError
		switch {
		case errors.As(err, &notFound):
			writeOpenAIError(w, http.StatusNotFound, fmt.Sprintf("The model '%s' does not exist", notFound.Ref), "model", "model_not_found")
		case errors.As(err, &ambiguous):
			msg := fmt.Sprintf("model '%s' is ambiguous; available tags: %s", ambiguous.Name, strings.Join(ambiguous.Tags, ", "))
			writeOpenAIError(w, http.StatusBadRequest, msg, "model", "model_ambiguous")
		default:
			writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "", "internal_error")
		}
		return
	}

	inst, err := s.getOrLaunchInstance(r.Context(), model)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "", "internal_error")
		return
	}

	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(inst.URL())
		},
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) {
				return
			}
			writeOpenAIError(w, http.StatusBadGateway, fmt.Sprintf("proxy error: %v", err), "", "bad_gateway")
		},
	}
	proxy.ServeHTTP(w, r)
}

func (s *Supervisor) getOrLaunchInstance(ctx context.Context, m Model) (host.Instance, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errors.New("supervisor is closed")
	}

	if inst, ok := s.instances[m.ID]; ok && inst != nil {
		select {
		case <-inst.Done():
			delete(s.instances, m.ID)
		default:
			s.mu.Unlock()
			return inst, nil
		}
	}

	if req, ok := s.loading[m.ID]; ok {
		s.mu.Unlock()
		select {
		case <-req.ready:
			if req.err != nil {
				return nil, req.err
			}
			return req.inst, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	req := &loadRequest{ready: make(chan struct{})}
	s.loading[m.ID] = req
	s.mu.Unlock()

	newInst, err := s.launchInstance(m)

	s.mu.Lock()
	delete(s.loading, m.ID)
	req.inst = newInst
	req.err = err

	if err == nil {
		if s.closed {
			stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = newInst.Stop(stopCtx)
			req.inst = nil
			req.err = errors.New("supervisor is closed")
		} else {
			s.instances[m.ID] = newInst
		}
	}
	close(req.ready)
	s.mu.Unlock()

	if req.err != nil {
		return nil, req.err
	}
	return req.inst, nil
}

func (s *Supervisor) launchInstance(m Model) (host.Instance, error) {
	ctxLen := m.ContextLength
	if ctxLen == 0 {
		ctxLen = 2048
	}
	argv := []string{
		"llama-server",
		"-m", m.Path,
		"-c", strconv.FormatUint(ctxLen, 10),
		"-np", "1",
	}

	newInst, err := s.host.Launch(context.Background(), argv)
	if err != nil {
		return nil, fmt.Errorf("failed to launch instance: %w", err)
	}

	readyCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := newInst.WaitReady(readyCtx); err != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer stopCancel()
		_ = newInst.Stop(stopCtx)
		return nil, fmt.Errorf("instance failed readiness: %w", err)
	}

	return newInst, nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
