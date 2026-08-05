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

type pendingInstance struct {
	ready  chan struct{}
	cancel context.CancelFunc
	inst   host.Instance
	err    error
}

func stopInstance(ctx context.Context, inst host.Instance, timeout time.Duration) error {
	if inst == nil {
		return nil
	}
	stopCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return inst.Stop(stopCtx)
}

// Supervisor is the daemon's root object. It owns Model discovery, the HTTP surface,
// and, in later work, the Instance registry and the Tuning cache.
type tunedConfig struct {
	CtxLen  uint64
	Offload uint64
}

type Supervisor struct {
	mu         sync.RWMutex
	host       host.Host
	models     map[string]Model
	modelsList []Model
	instances  map[string]host.Instance
	loading    map[string]*pendingInstance
	tuned      map[string]tunedConfig
	tuningMu   sync.Mutex
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
		loading:    make(map[string]*pendingInstance),
		tuned:      make(map[string]tunedConfig),
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
	if err != nil {
		return err
	}
	modelID := m.ID

	s.mu.Lock()
	inst, resident := s.instances[modelID]
	if resident {
		delete(s.instances, modelID)
	}
	req, loading := s.loading[modelID]
	if loading && req != nil && req.cancel != nil {
		req.cancel()
	}
	s.mu.Unlock()

	var stopErr error
	if resident {
		stopErr = stopInstance(ctx, inst, 10*time.Second)
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
				if err := stopInstance(ctx, loadedInst, 10*time.Second); err != nil && stopErr == nil {
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
	inFlight := make([]*pendingInstance, 0, len(s.loading))
	for _, req := range s.loading {
		inFlight = append(inFlight, req)
	}
	s.mu.Unlock()

	var errs []error
	for _, inst := range instancesToStop {
		if err := stopInstance(context.Background(), inst, 10*time.Second); err != nil {
			errs = append(errs, fmt.Errorf("stopping instance: %w", err))
		}
	}

	for _, req := range inFlight {
		select {
		case <-req.ready:
			if req.inst != nil {
				if err := stopInstance(context.Background(), req.inst, 10*time.Second); err != nil {
					errs = append(errs, fmt.Errorf("stopping in-flight instance: %w", err))
				}
			}
		case <-time.After(10 * time.Second):
			errs = append(errs, errors.New("timed out waiting for in-flight load to stop"))
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

	tags := make([]string, len(matches))
	for i, m := range matches {
		tags[i] = strings.ToLower(m.Quantization)
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
		if errors.Is(err, context.Canceled) {
			return
		}
		status := http.StatusInternalServerError
		code := "internal_error"
		if err.Error() == "supervisor is closed" {
			status = http.StatusServiceUnavailable
			code = "service_unavailable"
		}
		writeOpenAIError(w, status, err.Error(), "", code)
		return
	}

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = inst.URL().Scheme
			req.URL.Host = inst.URL().Host
		},
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) {
				return
			}
			writeOpenAIError(w, http.StatusServiceUnavailable, fmt.Sprintf("proxy error: %v", err), "", "service_unavailable")
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

	loadCtx, loadCancel := context.WithCancel(context.Background())
	req := &pendingInstance{
		ready:  make(chan struct{}),
		cancel: loadCancel,
	}
	s.loading[m.ID] = req
	s.mu.Unlock()

	newInst, err := s.launchInstance(loadCtx, m)

	s.mu.Lock()
	delete(s.loading, m.ID)
	req.inst = newInst
	req.err = err

	var instToStop host.Instance
	if err == nil {
		if s.closed {
			instToStop = newInst
			req.inst = nil
			req.err = errors.New("supervisor is closed")
		} else {
			s.instances[m.ID] = newInst
		}
	}
	close(req.ready)
	s.mu.Unlock()

	if instToStop != nil {
		_ = stopInstance(context.Background(), instToStop, 5*time.Second)
	}

	if req.err != nil {
		return nil, req.err
	}
	return req.inst, nil
}

func (s *Supervisor) evictAllResidentInstances() {
	s.mu.Lock()
	var toStop []host.Instance
	for id, inst := range s.instances {
		toStop = append(toStop, inst)
		delete(s.instances, id)
	}
	s.mu.Unlock()

	for _, inst := range toStop {
		_ = stopInstance(context.Background(), inst, 5*time.Second)
	}
}

func (s *Supervisor) launchConfig(loadCtx context.Context, m Model, cfg tunedConfig) (host.Instance, error) {
	argv := []string{
		"llama-server",
		"-m", m.Path,
		"-c", strconv.FormatUint(cfg.CtxLen, 10),
		"-ngl", strconv.FormatUint(cfg.Offload, 10),
		"-np", "1",
	}

	newInst, err := s.host.Launch(loadCtx, argv)
	if err != nil {
		return nil, fmt.Errorf("failed to launch instance: %w", err)
	}

	readyCtx, cancel := context.WithTimeout(loadCtx, 30*time.Second)
	defer cancel()

	if err := newInst.WaitReady(readyCtx); err != nil {
		_ = stopInstance(context.Background(), newInst, 5*time.Second)
		return nil, fmt.Errorf("instance failed readiness: %w", err)
	}

	return newInst, nil
}

func (s *Supervisor) tune(loadCtx context.Context, m Model) (host.Instance, error) {
	s.tuningMu.Lock()
	defer s.tuningMu.Unlock()

	s.evictAllResidentInstances()

	ctxLen := m.ContextLength
	if ctxLen == 0 {
		ctxLen = 2048
	}
	var ladder []uint64
	for _, step := range []uint64{ctxLen, 32768, 16384, 8192, 4096, 2048, 1024, 512} {
		if step <= ctxLen {
			if len(ladder) == 0 || ladder[len(ladder)-1] != step {
				ladder = append(ladder, step)
			}
		}
	}

	maxOffload := m.BlockCount
	if maxOffload == 0 {
		maxOffload = 100 // fallback
	}

	var lastErr error

	for _, currentCtx := range ladder {
		cfg := tunedConfig{CtxLen: currentCtx, Offload: maxOffload}
		inst, err := s.launchConfig(loadCtx, m, cfg)
		if err == nil {
			s.mu.Lock()
			s.tuned[m.ID] = cfg
			s.mu.Unlock()
			return inst, nil
		}
		lastErr = err
		if !host.IsOOM(err) {
			return nil, fmt.Errorf("tuning aborted due to non-memory error: %w", err)
		}

		low := 0
		high := int(maxOffload) - 1
		var bestInst host.Instance
		var bestCfg tunedConfig

		for low <= high {
			mid := low + (high-low)/2
			cfg = tunedConfig{CtxLen: currentCtx, Offload: uint64(mid)}
			inst, err = s.launchConfig(loadCtx, m, cfg)

			if err == nil {
				if bestInst != nil {
					_ = stopInstance(context.Background(), bestInst, 5*time.Second)
				}
				bestInst = inst
				bestCfg = cfg
				low = mid + 1
			} else {
				lastErr = err
				if !host.IsOOM(err) {
					if bestInst != nil {
						_ = stopInstance(context.Background(), bestInst, 5*time.Second)
					}
					return nil, fmt.Errorf("tuning aborted due to non-memory error: %w", err)
				}
				high = mid - 1
			}
		}

		if bestInst != nil {
			s.mu.Lock()
			s.tuned[m.ID] = bestCfg
			s.mu.Unlock()
			return bestInst, nil
		}
	}

	return nil, fmt.Errorf("model is unservable: exceeds available memory even at minimum configuration (last error: %v)", lastErr)
}

func (s *Supervisor) launchInstance(loadCtx context.Context, m Model) (host.Instance, error) {
	s.mu.RLock()
	cfg, ok := s.tuned[m.ID]
	s.mu.RUnlock()

	if !ok {
		return s.tune(loadCtx, m)
	}

	return s.launchConfig(loadCtx, m, cfg)
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
