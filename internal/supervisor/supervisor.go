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
	"log/slog"
	"net/http"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
)
var managedFlags = []struct {
	Short string
	Long  string
}{
	{"-m", "--model"},
	{"-c", "--ctx-size"},
	{"-ngl", "--n-gpu-layers"},
	{"-np", "--parallel"},
}

func isReservedFlag(arg string) bool {
	for _, f := range managedFlags {
		if arg == f.Short || strings.HasPrefix(arg, f.Short+"=") ||
			arg == f.Long || strings.HasPrefix(arg, f.Long+"=") {
			return true
		}
	}
	return false
}


type pendingInstance struct {
	mu      sync.Mutex
	ready   chan struct{}
	cancel  context.CancelFunc
	callers int
	inst    host.Instance
	err     error
}

type residentInstance struct {
	modelID     string
	inst        host.Instance
	ttl         time.Duration
	activeCount int
	maxSlots    int
	draining    bool
	stopped     bool
	idleTimer   *time.Timer
	timerActive bool
	lastUsed    time.Time
	mu          sync.Mutex
	idleCond    *sync.Cond
}

func newResidentInstance(modelID string, inst host.Instance, ttl time.Duration, maxSlots int) *residentInstance {
	if maxSlots <= 0 {
		maxSlots = 1
	}
	ri := &residentInstance{
		modelID:  modelID,
		inst:     inst,
		ttl:      ttl,
		maxSlots: maxSlots,
		lastUsed: time.Now(),
	}
	ri.idleCond = sync.NewCond(&ri.mu)
	return ri
}

func (ri *residentInstance) Occupancy() (active, max int) {
	ri.mu.Lock()
	defer ri.mu.Unlock()
	return ri.activeCount, ri.maxSlots
}

func (ri *residentInstance) getLastUsed() time.Time {
	ri.mu.Lock()
	defer ri.mu.Unlock()
	return ri.lastUsed
}

// neverExpires is the sentinel timestamp reported to Ollama clients for instances with infinite TTL.
var neverExpires = time.Date(2099, 12, 31, 23, 59, 59, 0, time.UTC)

func (ri *residentInstance) ExpiresAt() time.Time {
	ri.mu.Lock()
	defer ri.mu.Unlock()
	if ri.timerActive {
		return time.Now().Add(ri.ttl)
	}
	if ri.ttl > 0 {
		return ri.lastUsed.Add(ri.ttl)
	}
	return neverExpires
}

func (ri *residentInstance) acquire(ctx context.Context) bool {
	ri.mu.Lock()
	defer ri.mu.Unlock()

	if ctx.Err() != nil {
		return false
	}
	if ri.draining || ri.stopped {
		return false
	}
	select {
	case <-ri.inst.Done():
		return false
	default:
	}

	if ri.maxSlots <= 0 {
		ri.maxSlots = 1
	}

	if ri.activeCount >= ri.maxSlots {
		stopCtxWait := make(chan struct{})
		defer close(stopCtxWait)

		go func() {
			select {
			case <-ctx.Done():
				ri.mu.Lock()
				ri.idleCond.Broadcast()
				ri.mu.Unlock()
			case <-stopCtxWait:
			}
		}()

		for ri.activeCount >= ri.maxSlots {
			if ctx.Err() != nil {
				return false
			}
			if ri.draining || ri.stopped {
				return false
			}
			select {
			case <-ri.inst.Done():
				return false
			default:
			}

			ri.idleCond.Wait()
		}
	}

	if ctx.Err() != nil {
		return false
	}
	if ri.draining || ri.stopped {
		return false
	}
	select {
	case <-ri.inst.Done():
		return false
	default:
	}

	ri.activeCount++
	ri.lastUsed = time.Now()
	if ri.idleTimer != nil && ri.timerActive {
		ri.idleTimer.Stop()
		ri.timerActive = false
	}
	return true
}

func (ri *residentInstance) release(s *Supervisor) {
	ri.mu.Lock()
	defer ri.mu.Unlock()

	ri.lastUsed = time.Now()
	if ri.activeCount > 0 {
		ri.activeCount--
	}
	ri.idleCond.Broadcast()
	if ri.activeCount == 0 {
		if ri.draining {
			return
		}
		if ri.ttl > 0 && !ri.stopped {
			if ri.idleTimer != nil {
				ri.idleTimer.Stop()
			}
			ri.timerActive = true
			ri.idleTimer = time.AfterFunc(ri.ttl, func() {
				s.onInstanceIdleTimeout(ri)
			})
		}
	}
}

func (ri *residentInstance) stopTimerAndMarkStopped() {
	ri.mu.Lock()
	defer ri.mu.Unlock()

	if ri.idleTimer != nil {
		ri.idleTimer.Stop()
		ri.timerActive = false
	}
	ri.stopped = true
	ri.idleCond.Broadcast()
}

func (ri *residentInstance) drainAndStop(s *Supervisor, reason string) {
	ri.mu.Lock()
	if ri.stopped {
		ri.mu.Unlock()
		return
	}
	if ri.idleTimer != nil && ri.timerActive {
		ri.idleTimer.Stop()
		ri.timerActive = false
	}

	if ri.activeCount > 0 {
		ri.draining = true
		for ri.activeCount > 0 && !ri.stopped {
			ri.idleCond.Wait()
		}
	}

	if ri.stopped {
		ri.mu.Unlock()
		return
	}
	ri.stopped = true
	ri.mu.Unlock()

	s.mu.Lock()
	if current, ok := s.instances[ri.modelID]; ok && current == ri {
		delete(s.instances, ri.modelID)
	}
	s.mu.Unlock()

	slog.Info("resident instance stopped", "model", ri.modelID, "reason", reason)
	_ = stopInstance(context.Background(), ri.inst, 10*time.Second)
}

func (s *Supervisor) onInstanceIdleTimeout(ri *residentInstance) {
	ri.mu.Lock()
	if ri.stopped || !ri.timerActive {
		ri.mu.Unlock()
		return
	}
	ri.timerActive = false
	ri.mu.Unlock()

	ri.drainAndStop(s, "idle TTL expired")
}

func stopInstance(ctx context.Context, inst host.Instance, timeout time.Duration) error {
	if inst == nil {
		return nil
	}
	stopCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return inst.Stop(stopCtx)
}

// TuningEntry represents a cached, empirically measured tuning configuration for a Model.
type TuningEntry struct {
	ModelID      string          `json:"model_id"`
	ModelDigest  string          `json:"model_digest"`
	Fingerprint  string          `json:"fingerprint"`
	RequestedCtx uint64          `json:"requested_ctx"`
	KVCacheType  string          `json:"kv_cache_type"`
	Offload      uint64          `json:"offload"`
	ResultingCtx uint64          `json:"resulting_ctx"`
	Allocation   host.Allocation `json:"allocation"`
	MeasuredAt   time.Time       `json:"measured_at"`
}

// TuningCacheFile represents the root JSON structure persisted on disk.
type TuningCacheFile struct {
	Fingerprint string                 `json:"fingerprint"`
	Entries     map[string]TuningEntry `json:"entries"`
}

type tunedConfig struct {
	CtxLen  uint64
	Offload uint64
}

// TunedPin pins a Model's launch flags verbatim, bypassing empirical
// measurement entirely. It is never written to the Tuning cache file: it is
// a static override supplied by the operator, re-applied on every launch.
//
// Both fields are required together: json.Unmarshal leaves a missing field
// nil rather than zero, so loadConfig can reject a partial pin instead of
// silently launching with the other flag at its zero value (CPU-only
// offload, or an unusable zero-length context) with no measurement left to
// correct it.
type TunedPin struct {
	CtxLen  *uint64 `json:"ctx_len"`
	Offload *uint64 `json:"offload"`
}

// ModelConfig specifies configuration overrides for a specific Model. Every
// field is optional; an unset field falls through to the Supervisor's
// computed or global default.
type ModelConfig struct {
	// Argv is appended to the launch argv the Supervisor computes for this
	// Model (after -m/-c/-ngl/-np), letting an operator add flags llama-server
	// supports that the Supervisor has no opinion about.
	Argv []string `json:"argv,omitempty"`
	// TTL overrides the default idle TTL for this Model. Parsed with
	// time.ParseDuration (e.g. "5m", "30s").
	TTL string `json:"ttl,omitempty"`
	// Slots overrides the default Slot count for this Model.
	Slots *int `json:"slots,omitempty"`
	// Tuned pins this Model's context length and offload verbatim, skipping
	// measurement entirely.
	Tuned *TunedPin `json:"tuned,omitempty"`
}

// Config represents the optional configuration file's overrides. Config
// exists to override per-Model settings, never to enumerate Models:
// discovery alone determines what is servable, and an entry here is looked
// up by Model ID, name, path, or filename — never required.
type Config struct {
	Models map[string]ModelConfig `json:"models"`
}

// Option configures optional parameters on a Supervisor.
type Option func(*Supervisor)

// WithCachePath sets the file path for persisting tuning results.
func WithCachePath(path string) Option {
	return func(s *Supervisor) {
		s.cachePath = path
	}
}

// WithConfigFile sets the path to the optional configuration file. The file
// overrides per-Model argv, TTL, Slot count, and tuned values; its absence
// is not an error — Model discovery alone is enough to serve every Model.
func WithConfigFile(path string) Option {
	return func(s *Supervisor) {
		s.configPath = path
	}
}

// WithRescanInterval sets how often the Supervisor re-scans its configured
// directories in the background. A value <= 0 disables periodic rescanning;
// Rescan can still be triggered on demand.
func WithRescanInterval(d time.Duration) Option {
	return func(s *Supervisor) {
		s.rescanInterval = d
	}
}

// WithTuningBudget sets the maximum duration allowed for a single tuning run.
func WithTuningBudget(d time.Duration) Option {
	return func(s *Supervisor) {
		s.tuningBudget = d
	}
}

// WithDefaultTTL sets the default idle TTL for resident instances.
func WithDefaultTTL(d time.Duration) Option {
	return func(s *Supervisor) {
		s.defaultTTL = d
	}
}

// WithTTL is an alias for WithDefaultTTL.
func WithTTL(d time.Duration) Option {
	return WithDefaultTTL(d)
}

// WithModelTTL sets a per-model idle TTL override.
func WithModelTTL(modelRef string, d time.Duration) Option {
	return func(s *Supervisor) {
		if s.modelTTLs == nil {
			s.modelTTLs = make(map[string]time.Duration)
		}
		s.modelTTLs[modelRef] = d
	}
}

// WithMaxInstances sets the maximum number of resident instances allowed.
// A value <= 0 means unlimited resident instances (uncapped).
func WithMaxInstances(n int) Option {
	return func(s *Supervisor) {
		s.maxInstances = n
	}
}

// WithMaxResidentInstances is an alias for WithMaxInstances.
func WithMaxResidentInstances(n int) Option {
	return WithMaxInstances(n)
}

// WithSlotsPerInstance sets the number of concurrent slots per resident instance.
// A value <= 0 defaults to 1.
func WithSlotsPerInstance(n int) Option {
	return func(s *Supervisor) {
		if n <= 0 {
			n = 1
		}
		s.slotsPerInstance = n
	}
}

// WithSlots is an alias for WithSlotsPerInstance.
func WithSlots(n int) Option {
	return WithSlotsPerInstance(n)
}

// Supervisor is the daemon's root object. It owns Model discovery, the HTTP surface,
// the Instance registry, and the Tuning cache.
type Supervisor struct {
	mu                   sync.RWMutex
	host                 host.Host
	models               map[string]Model
	modelsList           []Model
	instances            map[string]*residentInstance
	loading              map[string]*pendingInstance
	tuned                map[string]TuningEntry
	tuningMu             sync.Mutex
	tuningActiveModel    string
	tuningStartedAt      time.Time
	tuningCurrentCtx     uint64
	tuningCurrentOffload uint64
	tuningProbeCount     int
	cachePath            string
	tuningBudget         time.Duration
	defaultTTL           time.Duration
	modelTTLs            map[string]time.Duration
	maxInstances         int
	slotsPerInstance     int
	configPath           string
	config               Config
	configTTL            map[string]time.Duration
	dirs                 []string
	rescanInterval       time.Duration
	rescanStop           chan struct{}
	rescanStopOnce       sync.Once
	closed               bool
}

// New builds a Supervisor by scanning the configured directories for Models.
// Host is the single injected boundary in the system representing physical hardware.
// If h is nil, a default real Host is used.
func New(h host.Host, dirs ...string) (*Supervisor, error) {
	return NewWithOpts(h, dirs)
}

// NewWithOpts builds a Supervisor with optional configuration options.
func NewWithOpts(h host.Host, dirs []string, opts ...Option) (*Supervisor, error) {
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

	defaultCachePath := "tuning.json"
	if len(dirs) > 0 && dirs[0] != "" {
		defaultCachePath = filepath.Join(dirs[0], "tuning.json")
	}
	defaultBudget := 2 * time.Minute

	s := &Supervisor{
		host:             h,
		models:           modelMap,
		modelsList:       models,
		instances:        make(map[string]*residentInstance),
		loading:          make(map[string]*pendingInstance),
		tuned:            make(map[string]TuningEntry),
		cachePath:        defaultCachePath,
		tuningBudget:     defaultBudget,
		defaultTTL:       5 * time.Minute,
		modelTTLs:        make(map[string]time.Duration),
		slotsPerInstance: 1,
		dirs:             dirs,
		rescanInterval:   30 * time.Second,
		rescanStop:       make(chan struct{}),
	}
	for _, opt := range opts {
		opt(s)
	}

	s.loadTuningCache()

	if s.configPath != "" {
		if err := s.loadConfig(); err != nil {
			return nil, err
		}
	}

	if s.rescanInterval > 0 {
		s.startRescanTimer()
	}

	return s, nil
}

// SlotsPerInstance returns the configured number of concurrent slots per resident instance.
func (s *Supervisor) SlotsPerInstance() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.slotsPerInstance
}

// InstanceOccupancy returns the current active and total slot count for a resident instance, if loaded.
func (s *Supervisor) InstanceOccupancy(modelID string) (active, max int, ok bool) {
	s.mu.RLock()
	resInst, exists := s.instances[modelID]
	s.mu.RUnlock()
	if !exists || resInst == nil {
		return 0, 0, false
	}
	active, max = resInst.Occupancy()
	return active, max, true
}

func (s *Supervisor) loadTuningCache() {
	if s.cachePath == "" {
		return
	}
	data, err := os.ReadFile(s.cachePath)
	if err != nil {
		return
	}

	var cacheFile TuningCacheFile
	if err := json.Unmarshal(data, &cacheFile); err != nil {
		slog.Warn("failed to parse tuning cache file, resetting", "path", s.cachePath, "err", err)
		_ = os.Remove(s.cachePath)
		return
	}

	fp := s.host.Fingerprint()
	if cacheFile.Fingerprint != fp {
		slog.Info("hardware fingerprint changed, invalidating tuning cache", "old", cacheFile.Fingerprint, "new", fp)
		_ = os.Remove(s.cachePath)
		return
	}

	for id, entry := range cacheFile.Entries {
		if entry.Fingerprint == fp {
			s.tuned[id] = entry
		}
	}
}

func (s *Supervisor) saveTuningCacheLocked() {
	if s.cachePath == "" {
		return
	}
	cacheFile := TuningCacheFile{
		Fingerprint: s.host.Fingerprint(),
		Entries:     s.tuned,
	}
	data, err := json.MarshalIndent(cacheFile, "", "  ")
	if err != nil {
		slog.Error("failed to marshal tuning cache", "err", err)
		return
	}

	dir := filepath.Dir(s.cachePath)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			slog.Error("failed to create directory for tuning cache", "dir", dir, "err", err)
			return
		}
	}

	tmpPath := s.cachePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		slog.Error("failed to write tuning cache temp file", "tmpPath", tmpPath, "err", err)
		return
	}
	if err := os.Rename(tmpPath, s.cachePath); err != nil {
		slog.Error("failed to rename tuning cache file", "err", err)
	}
}

// loadConfig reads and validates the optional configuration file at
// s.configPath. Model overrides are looked up lazily against live Model
// data (see resolveModelConfig), so a config entry for a Model that has not
// been discovered yet — or is added by a later Rescan — is not an error.
func (s *Supervisor) loadConfig() error {
	data, err := os.ReadFile(s.configPath)
	if err != nil {
		return fmt.Errorf("reading config file %q: %w", s.configPath, err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parsing config file %q: %w", s.configPath, err)
	}

	configTTL := make(map[string]time.Duration, len(cfg.Models))
	for ref, mc := range cfg.Models {
		if mc.TTL != "" {
			d, err := time.ParseDuration(mc.TTL)
			if err != nil {
				return fmt.Errorf("config file %q: model %q: invalid ttl %q: %w", s.configPath, ref, mc.TTL, err)
			}
			configTTL[ref] = d
		}
		for _, arg := range mc.Argv {
			if isReservedFlag(arg) {
				return fmt.Errorf("config file %q: model %q: argv cannot override reserved flag %q", s.configPath, ref, arg)
			}
		}
		if mc.Slots != nil && *mc.Slots <= 0 {
			return fmt.Errorf("config file %q: model %q: slots must be positive, got %d", s.configPath, ref, *mc.Slots)
		}
		if mc.Tuned != nil {
			if mc.Tuned.CtxLen == nil || mc.Tuned.Offload == nil {
				return fmt.Errorf("config file %q: model %q: tuned requires both ctx_len and offload", s.configPath, ref)
			}
			if *mc.Tuned.CtxLen == 0 {
				return fmt.Errorf("config file %q: model %q: tuned ctx_len must be positive", s.configPath, ref)
			}
			// Note: *mc.Tuned.Offload == 0 is deliberately legal (CPU-only pin).
		}
	}

	s.config = cfg
	s.configTTL = configTTL
	return nil
}

// lookupByModelKeys checks m against ID, then Name, then Path, then base
// filename, in that order — the precedence every config and runtime
// override in this file uses, since an operator may key an override by
// whichever form of a Model's identity they have on hand.
func lookupByModelKeys[V any](m Model, byKey map[string]V) (V, bool) {
	if v, ok := byKey[m.ID]; ok {
		return v, true
	}
	if v, ok := byKey[m.Name]; ok {
		return v, true
	}
	if v, ok := byKey[m.Path]; ok {
		return v, true
	}
	if v, ok := byKey[filepath.Base(m.Path)]; ok {
		return v, true
	}
	var zero V
	return zero, false
}

// resolveModelConfig looks up m's configuration override. s.config is
// populated once at startup and never mutated afterward, so this is safe
// to call without holding s.mu.
func (s *Supervisor) resolveModelConfig(m Model) (ModelConfig, bool) {
	return lookupByModelKeys(m, s.config.Models)
}

// resolveConfigTTL returns m's configured TTL override, pre-parsed and
// validated once by loadConfig rather than re-parsed on every call.
func (s *Supervisor) resolveConfigTTL(m Model) (time.Duration, bool) {
	return lookupByModelKeys(m, s.configTTL)
}

// resolveSlots returns m's configured Slot count, falling back to the
// Supervisor-wide default.
func (s *Supervisor) resolveSlots(m Model) int {
	if mc, ok := s.resolveModelConfig(m); ok && mc.Slots != nil {
		return *mc.Slots
	}
	return s.slotsPerInstance
}

// resolveArgv returns extra argv appended to m's computed launch command.
func (s *Supervisor) resolveArgv(m Model) []string {
	if mc, ok := s.resolveModelConfig(m); ok {
		return mc.Argv
	}
	return nil
}

// resolveTunedOverride returns m's pinned Tuned configuration, if the
// operator configured one. A pinned value bypasses measurement entirely and
// is never written to the Tuning cache file. loadConfig already rejected
// any config where Tuned is set but CtxLen or Offload is nil, so both
// pointers are safe to dereference here.
func (s *Supervisor) resolveTunedOverride(m Model) (tunedConfig, bool) {
	if mc, ok := s.resolveModelConfig(m); ok && mc.Tuned != nil {
		return tunedConfig{CtxLen: *mc.Tuned.CtxLen, Offload: *mc.Tuned.Offload}, true
	}
	return tunedConfig{}, false
}

// Rescan re-scans the Supervisor's configured directories for Models,
// picking up newly added or removed GGUF files. It never restarts resident
// Instances or requires a restart of the Supervisor itself.
//
// A scan that comes back empty after Models were already being served is
// treated as a transient discovery failure (an unmounted drive, a network
// share blip) rather than an instruction to stop serving everything: it is
// logged and the existing Model list is left in place. Startup already
// refuses to begin with zero Models, so an operator who genuinely empties
// every scanned directory only sees that reflected after a restart.
func (s *Supervisor) Rescan() error {
	s.mu.RLock()
	dirs := s.dirs
	closed := s.closed
	hadModels := len(s.modelsList) > 0
	s.mu.RUnlock()
	if closed {
		return errors.New("supervisor is closed")
	}

	models, err := discoverModels(dirs)
	if err != nil {
		return err
	}

	if len(models) == 0 && hadModels {
		slog.Warn("rescan found no models, keeping previous model list", "dirs", dirs)
		return nil
	}

	modelMap := make(map[string]Model, len(models))
	for _, m := range models {
		modelMap[m.ID] = m
	}

	s.mu.Lock()
	s.models = modelMap
	s.modelsList = models
	s.mu.Unlock()
	return nil
}

// startRescanTimer launches the background goroutine that calls Rescan on
// s.rescanInterval, until Close stops it.
func (s *Supervisor) startRescanTimer() {
	ticker := time.NewTicker(s.rescanInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := s.Rescan(); err != nil {
					slog.Warn("periodic rescan failed", "err", err)
				}
			case <-s.rescanStop:
				return
			}
		}
	}()
}

// stopRescanTimer signals the background rescan goroutine to exit. Safe to
// call multiple times or when no timer was started.
func (s *Supervisor) stopRescanTimer() {
	s.rescanStopOnce.Do(func() { close(s.rescanStop) })
}

// Handler returns the Supervisor's HTTP router. This is the single router the
// binary serves and the one tests drive; there is no separate test wiring.
func (s *Supervisor) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /api/tags", s.handleAPITags)
	mux.HandleFunc("POST /api/chat", s.handleAPIChat)
	mux.HandleFunc("POST /api/generate", s.handleAPIGenerate)
	mux.HandleFunc("POST /api/embed", s.handleAPIEmbed)
	mux.HandleFunc("POST /api/embeddings", s.handleAPIEmbed)
	mux.HandleFunc("GET /api/version", s.handleAPIVersion)
	mux.HandleFunc("POST /api/show", s.handleAPIShow)
	mux.HandleFunc("GET /api/ps", s.handleAPIPs)
	mux.HandleFunc("GET /v1/models", s.handleV1Models)
	mux.HandleFunc("POST /v1/chat/completions", s.handleV1OpenAIProxy)
	mux.HandleFunc("POST /v1/completions", s.handleV1OpenAIProxy)
	mux.HandleFunc("POST /v1/embeddings", s.handleV1OpenAIProxy)
	mux.HandleFunc("GET /v1/tuning", s.handleV1TuningGet)
	mux.HandleFunc("POST /v1/tuning/reset", s.handleV1TuningReset)
	mux.HandleFunc("DELETE /v1/tuning", s.handleV1TuningReset)
	mux.HandleFunc("GET /api/tuning", s.handleV1TuningGet)
	mux.HandleFunc("POST /api/tuning/reset", s.handleV1TuningReset)
	mux.HandleFunc("DELETE /api/tuning", s.handleV1TuningReset)
	mux.HandleFunc("POST /v1/rescan", s.handleV1Rescan)
	mux.HandleFunc("POST /api/rescan", s.handleV1Rescan)
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
	resInst, resident := s.instances[modelID]
	if resident {
		delete(s.instances, modelID)
		resInst.stopTimerAndMarkStopped()
	}
	req, loading := s.loading[modelID]
	if loading && req != nil && req.cancel != nil {
		req.cancel()
	}
	s.mu.Unlock()

	var stopErr error
	if resident {
		stopErr = stopInstance(ctx, resInst.inst, 10*time.Second)
	}

	if loading && req != nil {
		select {
		case <-req.ready:
			s.mu.Lock()
			loadedResInst, loadedOk := s.instances[modelID]
			if loadedOk {
				delete(s.instances, modelID)
				loadedResInst.stopTimerAndMarkStopped()
			}
			s.mu.Unlock()
			if loadedOk && loadedResInst != nil {
				if err := stopInstance(ctx, loadedResInst.inst, 10*time.Second); err != nil && stopErr == nil {
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
	s.stopRescanTimer()

	s.mu.Lock()
	s.closed = true
	instancesToStop := make([]host.Instance, 0, len(s.instances))
	for _, resInst := range s.instances {
		resInst.stopTimerAndMarkStopped()
		instancesToStop = append(instancesToStop, resInst.inst)
	}
	s.instances = make(map[string]*residentInstance)
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

func modelDetails(m Model) ollamaModelDetails {
	var families any
	if m.Architecture != "" {
		families = []string{m.Architecture}
	}
	return ollamaModelDetails{
		Format:            "gguf",
		Family:            m.Architecture,
		Families:          families,
		ParameterSize:     "",
		QuantizationLevel: m.Quantization,
	}
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
			Details:    modelDetails(m),
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

type openAIRequest struct {
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

	var m Model
	if exact, ok := s.models[ref]; ok {
		m = exact
	} else {
		var matches []Model
		for _, candidate := range s.modelsList {
			if candidate.ID == ref || candidate.Name == ref {
				matches = append(matches, candidate)
			}
		}

		if len(matches) == 0 {
			return Model{}, &ModelNotFoundError{Ref: ref}
		}

		if len(matches) == 1 {
			m = matches[0]
		} else {
			tags := make([]string, len(matches))
			for i, cand := range matches {
				tags[i] = cand.Tag
			}
			return Model{}, &AmbiguousModelError{Name: ref, Tags: tags}
		}
	}

	if info, err := os.Stat(m.Path); err == nil {
		if info.Size() != m.Size || !info.ModTime().Equal(m.ModTime) {
			m.Size = info.Size()
			m.ModTime = info.ModTime()
			m.Digest = computeDigest(m)
		}
	}

	return m, nil
}

func (s *Supervisor) handleV1OpenAIProxy(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "failed to read request body", "", "invalid_request")
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req openAIRequest
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

	inst, release, err := s.getOrLaunchInstance(r.Context(), model)
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
	defer release()

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

func (s *Supervisor) resolveTTLLocked(modelID string) time.Duration {
	if ttl, ok := s.modelTTLs[modelID]; ok {
		return ttl
	}
	if m, ok := s.models[modelID]; ok {
		if ttl, ok := lookupByModelKeys(m, s.modelTTLs); ok {
			return ttl
		}
		if ttl, ok := s.resolveConfigTTL(m); ok {
			return ttl
		}
	}
	return s.defaultTTL
}

// SetModelTTL sets a per-model idle TTL override at runtime.
func (s *Supervisor) SetModelTTL(modelRef string, d time.Duration) error {
	m, err := s.resolveModel(modelRef)
	if err != nil {
		return err
	}

	s.mu.Lock()
	if s.modelTTLs == nil {
		s.modelTTLs = make(map[string]time.Duration)
	}
	s.modelTTLs[m.ID] = d

	ri, resident := s.instances[m.ID]
	s.mu.Unlock()

	if resident && ri != nil {
		ri.mu.Lock()
		ri.ttl = d
		if ri.activeCount == 0 && !ri.draining && !ri.stopped {
			if ri.idleTimer != nil {
				ri.idleTimer.Stop()
				ri.timerActive = false
			}
			if d > 0 {
				ri.timerActive = true
				ri.idleTimer = time.AfterFunc(d, func() {
					s.onInstanceIdleTimeout(ri)
				})
			}
		}
		ri.mu.Unlock()
	}

	return nil
}
func (s *Supervisor) GetModelTTL(modelRef string) (time.Duration, error) {
	m, err := s.resolveModel(modelRef)
	if err != nil {
		return 0, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.resolveTTLLocked(m.ID), nil
}

func (s *Supervisor) getOrLaunchInstance(ctx context.Context, m Model) (host.Instance, func(), error) {
	for {
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			return nil, func() {}, errors.New("supervisor is closed")
		}

		if resInst, ok := s.instances[m.ID]; ok && resInst != nil {
			select {
			case <-resInst.inst.Done():
				resInst.stopTimerAndMarkStopped()
				delete(s.instances, m.ID)
			default:
				s.mu.Unlock()
				if resInst.acquire(ctx) {
					return resInst.inst, func() { resInst.release(s) }, nil
				}
				if ctx.Err() != nil {
					return nil, func() {}, ctx.Err()
				}
				s.mu.Lock()
				if current, ok := s.instances[m.ID]; ok && current == resInst {
					delete(s.instances, m.ID)
				}
				s.mu.Unlock()
				continue
			}
		}

		if req, ok := s.loading[m.ID]; ok {
			req.mu.Lock()
			req.callers++
			req.mu.Unlock()
			s.mu.Unlock()

			doneMonitor := make(chan struct{})
			defer close(doneMonitor)
			go func() {
				select {
				case <-ctx.Done():
					req.mu.Lock()
					req.callers--
					if req.callers == 0 && req.cancel != nil {
						req.cancel()
					}
					req.mu.Unlock()
				case <-req.ready:
				case <-doneMonitor:
				}
			}()

			select {
			case <-req.ready:
				if req.err != nil {
					return nil, func() {}, req.err
				}
				s.mu.Lock()
				resInst, ok := s.instances[m.ID]
				s.mu.Unlock()
				if ok && resInst != nil {
					if resInst.acquire(ctx) {
						return resInst.inst, func() { resInst.release(s) }, nil
					}
					if ctx.Err() != nil {
						return nil, func() {}, ctx.Err()
					}
					continue
				}
				if req.inst != nil {
					return req.inst, func() {}, nil
				}
				return nil, func() {}, errors.New("instance unavailable after load")
			case <-ctx.Done():
				return nil, func() {}, ctx.Err()
			}
		}

		loadCtx, loadCancel := context.WithCancel(context.Background())
		req := &pendingInstance{
			ready:   make(chan struct{}),
			cancel:  loadCancel,
			callers: 1,
		}
		s.loading[m.ID] = req
		s.mu.Unlock()

		doneMonitor := make(chan struct{})
		defer close(doneMonitor)
		go func() {
			select {
			case <-ctx.Done():
				req.mu.Lock()
				req.callers--
				if req.callers == 0 && req.cancel != nil {
					req.cancel()
				}
				req.mu.Unlock()
			case <-req.ready:
			case <-doneMonitor:
			}
		}()

		newInst, err := s.launchInstance(loadCtx, m)

		s.mu.Lock()
		delete(s.loading, m.ID)
		req.inst = newInst
		req.err = err

		var instToStop host.Instance
		var resInst *residentInstance
		if err == nil {
			if s.closed {
				instToStop = newInst
				req.inst = nil
				req.err = errors.New("supervisor is closed")
			} else {
				if s.maxInstances > 0 && len(s.instances) >= s.maxInstances {
					s.evictLRULocked()
				}
				ttl := s.resolveTTLLocked(m.ID)
				resInst = newResidentInstance(m.ID, newInst, ttl, s.resolveSlots(m))
				if !resInst.acquire(ctx) {
					instToStop = newInst
					req.inst = nil
					if ctx.Err() != nil {
						req.err = ctx.Err()
					} else {
						req.err = errors.New("instance unavailable after launch")
					}
				} else {
					s.instances[m.ID] = resInst
				}
			}
		}
		close(req.ready)
		s.mu.Unlock()

		if instToStop != nil {
			_ = stopInstance(context.Background(), instToStop, 5*time.Second)
		}

		if req.err != nil {
			return nil, func() {}, req.err
		}
		return resInst.inst, func() { resInst.release(s) }, nil
	}
}
func (s *Supervisor) evictLRULocked() {
	var lruID string
	var lruInst *residentInstance
	var oldest time.Time

	for id, ri := range s.instances {
		last := ri.getLastUsed()
		if lruInst == nil || last.Before(oldest) {
			oldest = last
			lruID = id
			lruInst = ri
		}
	}

	if lruInst != nil {
		delete(s.instances, lruID)
		lruInst.mu.Lock()
		lruInst.draining = true
		if lruInst.idleTimer != nil && lruInst.timerActive {
			lruInst.idleTimer.Stop()
			lruInst.timerActive = false
		}
		lruInst.mu.Unlock()
		go lruInst.drainAndStop(s, "evicted by LRU cap")
	}
}

func (s *Supervisor) evictAllResidentInstances() {
	s.mu.Lock()
	var toStop []host.Instance
	for id, resInst := range s.instances {
		resInst.stopTimerAndMarkStopped()
		toStop = append(toStop, resInst.inst)
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
		managedFlags[0].Short, m.Path,
		managedFlags[1].Short, strconv.FormatUint(cfg.CtxLen, 10),
		managedFlags[2].Short, strconv.FormatUint(cfg.Offload, 10),
		managedFlags[3].Short, strconv.Itoa(s.resolveSlots(m)),
	}
	argv = append(argv, s.resolveArgv(m)...)

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

	now := time.Now().UTC()
	s.mu.Lock()
	s.tuningActiveModel = m.ID
	s.tuningStartedAt = now
	s.tuningCurrentCtx = 0
	s.tuningCurrentOffload = 0
	s.tuningProbeCount = 0
	budget := s.tuningBudget
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.tuningActiveModel = ""
		s.tuningStartedAt = time.Time{}
		s.tuningCurrentCtx = 0
		s.tuningCurrentOffload = 0
		s.tuningProbeCount = 0
		s.mu.Unlock()
	}()

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

	budgetCtx, cancel := context.WithTimeout(loadCtx, budget)
	defer cancel()

	fallback := func(reason string) (host.Instance, error) {
		slog.Warn("tuning budget exhausted, falling back to conservative known-safe configuration",
			"model", m.ID,
			"budget", budget,
			"reason", reason,
		)

		fallbackCtx := uint64(2048)
		if ctxLen < fallbackCtx {
			fallbackCtx = ctxLen
		}
		fallbackCfg := tunedConfig{CtxLen: fallbackCtx, Offload: 0}

		launchCtx := loadCtx
		if launchCtx.Err() != nil {
			var launchCancel context.CancelFunc
			launchCtx, launchCancel = context.WithTimeout(context.Background(), 30*time.Second)
			defer launchCancel()
		}

		finalInst, err := s.launchConfig(launchCtx, m, fallbackCfg)
		if err != nil {
			return nil, fmt.Errorf("failed to launch conservative fallback configuration after budget exhaustion: %w", err)
		}

		entry := TuningEntry{
			ModelID:      m.ID,
			ModelDigest:  m.Digest,
			Fingerprint:  s.host.Fingerprint(),
			RequestedCtx: ctxLen,
			KVCacheType:  "f16",
			Offload:      fallbackCfg.Offload,
			ResultingCtx: fallbackCfg.CtxLen,
			Allocation:   finalInst.ObservedAllocation(),
			MeasuredAt:   time.Now().UTC(),
		}
		s.mu.Lock()
		s.tuned[m.ID] = entry
		s.saveTuningCacheLocked()
		s.mu.Unlock()

		return finalInst, nil
	}

	var lastErr error

	for _, currentCtx := range ladder {
		if budgetCtx.Err() != nil {
			if loadCtx.Err() != nil {
				return nil, loadCtx.Err()
			}
			return fallback(budgetCtx.Err().Error())
		}

		cfg := tunedConfig{CtxLen: currentCtx, Offload: maxOffload}
		s.mu.Lock()
		s.tuningCurrentCtx = cfg.CtxLen
		s.tuningCurrentOffload = cfg.Offload
		s.tuningProbeCount++
		s.mu.Unlock()

		inst, err := s.launchConfig(budgetCtx, m, cfg)
		if err == nil {
			entry := TuningEntry{
				ModelID:      m.ID,
				ModelDigest:  m.Digest,
				Fingerprint:  s.host.Fingerprint(),
				RequestedCtx: ctxLen,
				KVCacheType:  "f16",
				Offload:      cfg.Offload,
				ResultingCtx: cfg.CtxLen,
				Allocation:   inst.ObservedAllocation(),
				MeasuredAt:   time.Now().UTC(),
			}
			s.mu.Lock()
			s.tuned[m.ID] = entry
			s.saveTuningCacheLocked()
			s.mu.Unlock()
			return inst, nil
		}
		lastErr = err

		if budgetCtx.Err() != nil {
			if loadCtx.Err() != nil {
				return nil, loadCtx.Err()
			}
			return fallback(budgetCtx.Err().Error())
		}

		if !host.IsOOM(err) {
			return nil, fmt.Errorf("tuning aborted due to non-memory error: %w", err)
		}

		low := 0
		high := int(maxOffload) - 1
		var bestCfg tunedConfig
		var found bool

		for low <= high {
			if budgetCtx.Err() != nil {
				if loadCtx.Err() != nil {
					return nil, loadCtx.Err()
				}
				return fallback(budgetCtx.Err().Error())
			}

			mid := low + (high-low)/2
			cfg = tunedConfig{CtxLen: currentCtx, Offload: uint64(mid)}
			s.mu.Lock()
			s.tuningCurrentCtx = cfg.CtxLen
			s.tuningCurrentOffload = cfg.Offload
			s.tuningProbeCount++
			s.mu.Unlock()

			probeInst, err := s.launchConfig(budgetCtx, m, cfg)

			if err == nil {
				bestCfg = cfg
				found = true
				_ = stopInstance(context.Background(), probeInst, 5*time.Second)
				low = mid + 1
			} else {
				lastErr = err
				if budgetCtx.Err() != nil {
					if loadCtx.Err() != nil {
						return nil, loadCtx.Err()
					}
					return fallback(budgetCtx.Err().Error())
				}
				if !host.IsOOM(err) {
					return nil, fmt.Errorf("tuning aborted due to non-memory error: %w", err)
				}
				high = mid - 1
			}
		}

		if found {
			s.mu.Lock()
			s.tuningCurrentCtx = bestCfg.CtxLen
			s.tuningCurrentOffload = bestCfg.Offload
			s.tuningProbeCount++
			s.mu.Unlock()

			finalInst, err := s.launchConfig(budgetCtx, m, bestCfg)
			if err != nil {
				if budgetCtx.Err() != nil {
					if loadCtx.Err() != nil {
						return nil, loadCtx.Err()
					}
					return fallback(budgetCtx.Err().Error())
				}
				return nil, fmt.Errorf("failed to launch best tuned configuration: %w", err)
			}
			entry := TuningEntry{
				ModelID:      m.ID,
				ModelDigest:  m.Digest,
				Fingerprint:  s.host.Fingerprint(),
				RequestedCtx: ctxLen,
				KVCacheType:  "f16",
				Offload:      bestCfg.Offload,
				ResultingCtx: bestCfg.CtxLen,
				Allocation:   finalInst.ObservedAllocation(),
				MeasuredAt:   time.Now().UTC(),
			}
			s.mu.Lock()
			s.tuned[m.ID] = entry
			s.saveTuningCacheLocked()
			s.mu.Unlock()
			return finalInst, nil
		}
	}

	return nil, fmt.Errorf("model is unservable: exceeds available memory even at minimum configuration (last error: %v)", lastErr)
}

func (s *Supervisor) launchInstance(loadCtx context.Context, m Model) (host.Instance, error) {
	if cfg, ok := s.resolveTunedOverride(m); ok {
		return s.launchConfig(loadCtx, m, cfg)
	}

	s.mu.RLock()
	entry, ok := s.tuned[m.ID]
	s.mu.RUnlock()

	reqCtx := m.ContextLength
	if reqCtx == 0 {
		reqCtx = 2048
	}

	fp := s.host.Fingerprint()

	if ok {
		if entry.Fingerprint == fp &&
			entry.ModelDigest == m.Digest &&
			entry.RequestedCtx == reqCtx &&
			entry.KVCacheType == "f16" {
			cfg := tunedConfig{CtxLen: entry.ResultingCtx, Offload: entry.Offload}
			return s.launchConfig(loadCtx, m, cfg)
		}
		s.mu.Lock()
		delete(s.tuned, m.ID)
		s.saveTuningCacheLocked()
		s.mu.Unlock()
	}

	return s.tune(loadCtx, m)
}

// TuningProgress describes in-flight tuning status for an active model measurement.
type TuningProgress struct {
	StartedAt      time.Time `json:"started_at"`
	ElapsedSeconds float64   `json:"elapsed_seconds"`
	CurrentCtx     uint64    `json:"current_ctx"`
	CurrentOffload uint64    `json:"current_offload"`
	ProbeCount     int       `json:"probe_count"`
}

type tuningResponse struct {
	Status      string                 `json:"status"`
	ActiveModel string                 `json:"active_model,omitempty"`
	Fingerprint string                 `json:"fingerprint"`
	Progress    *TuningProgress        `json:"progress,omitempty"`
	Entries     map[string]TuningEntry `json:"entries"`
}

func (s *Supervisor) handleV1TuningGet(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	status := "idle"
	var prog *TuningProgress
	if s.tuningActiveModel != "" {
		status = "tuning"
		prog = &TuningProgress{
			StartedAt:      s.tuningStartedAt,
			ElapsedSeconds: time.Since(s.tuningStartedAt).Seconds(),
			CurrentCtx:     s.tuningCurrentCtx,
			CurrentOffload: s.tuningCurrentOffload,
			ProbeCount:     s.tuningProbeCount,
		}
	}

	resp := tuningResponse{
		Status:      status,
		ActiveModel: s.tuningActiveModel,
		Fingerprint: s.host.Fingerprint(),
		Progress:    prog,
		Entries:     make(map[string]TuningEntry, len(s.tuned)),
	}
	for id, entry := range s.tuned {
		resp.Entries[id] = entry
	}

	writeJSON(w, http.StatusOK, resp)
}

type tuningResetRequest struct {
	Model string `json:"model"`
}

func (s *Supervisor) handleV1TuningReset(w http.ResponseWriter, r *http.Request) {
	ref := r.URL.Query().Get("model")
	if ref == "" {
		ref = r.URL.Query().Get("ref")
	}

	if ref == "" && r.Body != nil && r.ContentLength > 0 {
		var req tuningResetRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		ref = req.Model
	}

	if ref != "" {
		m, err := s.resolveModel(ref)
		targetID := ref
		if err == nil {
			targetID = m.ID
		}

		_ = s.Evict(r.Context(), targetID)

		s.mu.Lock()
		delete(s.tuned, targetID)
		s.saveTuningCacheLocked()
		s.mu.Unlock()

		writeJSON(w, http.StatusOK, map[string]string{
			"status":  "ok",
			"message": fmt.Sprintf("tuning cache entry reset for model %s", targetID),
		})
		return
	}

	s.evictAllResidentInstances()

	s.mu.Lock()
	s.tuned = make(map[string]TuningEntry)
	if s.cachePath != "" {
		_ = os.Remove(s.cachePath)
	}
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"message": "all tuning cache entries reset",
	})
}

// rescanResponse reports the outcome of an on-demand directory rescan.
type rescanResponse struct {
	Status string `json:"status"`
	Models int    `json:"models"`
}

// handleV1Rescan triggers an on-demand re-scan of the Supervisor's
// configured directories, so a newly dropped GGUF becomes servable without
// a restart.
func (s *Supervisor) handleV1Rescan(w http.ResponseWriter, r *http.Request) {
	if err := s.Rescan(); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "error", "message": err.Error()})
		return
	}
	s.mu.RLock()
	n := len(s.modelsList)
	s.mu.RUnlock()
	writeJSON(w, http.StatusOK, rescanResponse{Status: "ok", Models: n})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
