package plugins

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

var namePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,23}$`)
var versionPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

type entry struct {
	ready       chan struct{}
	admin       sync.Mutex
	def         Definition
	settings    Settings
	status, err string
	ctx         context.Context
	cancel      context.CancelFunc
	active      int
	idle        chan struct{}
	jobs        map[string]context.CancelFunc
}
type Manager struct {
	catalog     sync.Mutex
	packageRoot string
	packages    map[string]PackageRecord
	mu          sync.Mutex
	store       Store
	services    map[string]Service
	entries     map[string]*entry
	closed      bool
}

func clone[T any](x T) T               { b, _ := json.Marshal(x); var y T; _ = json.Unmarshal(b, &y); return y }
func validJSON(b json.RawMessage) bool { return len(b) > 0 && len(b) <= MaxPayload && json.Valid(b) }
func object(b json.RawMessage) bool {
	return validJSON(b) && strings.HasPrefix(strings.TrimSpace(string(b)), "{")
}
func protect(fn func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("plugin panic: %v", p)
		}
	}()
	return fn()
}
func ToolName(id, name string) string { return "plugin__" + id + "__" + name }
func objectSchema(raw json.RawMessage) bool {
	if !object(raw) {
		return false
	}
	var schema struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &schema) == nil && schema.Type == "object"
}
func validate(d Definition, services map[string]Service) error {
	m := d.Manifest
	if !namePattern.MatchString(m.ID) || m.Name == "" || !versionPattern.MatchString(m.Version) || m.API != APIVersion || m.DataVersion < 1 || d.Handle == nil {
		return fmt.Errorf("invalid or incompatible plugin manifest: %s", m.ID)
	}
	perms := map[string]bool{}
	for _, p := range m.Permissions {
		if perms[p] {
			return fmt.Errorf("duplicate permission: %s", p)
		}
		if p != "storage" && p != "tools" && p != "jobs" && services[p] == nil {
			return fmt.Errorf("unknown permission: %s", p)
		}
		perms[p] = true
	}
	ops := map[string]bool{}
	for _, o := range m.Operations {
		if !namePattern.MatchString(o.Name) || ops[o.Name] || !objectSchema(o.Parameters) || o.TimeoutSeconds < 1 || o.TimeoutSeconds > 3600 {
			return fmt.Errorf("invalid operation: %s", o.Name)
		}
		if o.Tool && !perms["tools"] || o.Background && !perms["jobs"] {
			return fmt.Errorf("operation lacks declared permission: %s", o.Name)
		}
		ops[o.Name] = true
	}
	panels := map[string]bool{}
	for _, p := range m.Panels {
		if !namePattern.MatchString(p.ID) || p.Title == "" || panels[p.ID] {
			return fmt.Errorf("invalid panel")
		}
		panels[p.ID] = true
		for _, o := range p.Operations {
			if !ops[o] {
				return fmt.Errorf("unknown panel operation")
			}
		}
	}
	return nil
}
func New(ctx context.Context, store Store, services map[string]Service, defs []Definition) (*Manager, error) {
	m := &Manager{packages: map[string]PackageRecord{}, store: store, services: map[string]Service{}, entries: map[string]*entry{}}
	for k, v := range services {
		if k == "storage" || k == "tools" || k == "jobs" || v == nil {
			return nil, fmt.Errorf("invalid service: %s", k)
		}
		m.services[k] = v
	}
	toolNames := map[string]bool{}
	for _, d := range defs {
		if err := validate(d, m.services); err != nil {
			return nil, err
		}
		id := d.Manifest.ID
		if m.entries[id] != nil {
			return nil, fmt.Errorf("duplicate plugin: %s", id)
		}
		for _, op := range d.Manifest.Operations {
			if op.Tool {
				name := ToolName(id, op.Name)
				if toolNames[name] {
					return nil, fmt.Errorf("duplicate tool name: %s", name)
				}
				toolNames[name] = true
			}
		}
		d.Manifest = clone(d.Manifest)
		s, err := store.PluginSettings(ctx, id)
		if err != nil {
			return nil, err
		}
		if s.DataVersion == 0 {
			s = Settings{Config: json.RawMessage(`{}`), Grants: []string{}, DataVersion: d.Manifest.DataVersion}
			if err = store.SavePluginSettings(ctx, id, s); err != nil {
				return nil, err
			}
		}
		// A new manifest can remove capabilities; persisted grants must not
		// retain authority that the currently installed version no longer declares.
		declared := map[string]bool{}
		for _, permission := range d.Manifest.Permissions {
			declared[permission] = true
		}
		filtered := []string{}
		seen := map[string]bool{}
		for _, permission := range s.Grants {
			if declared[permission] && !seen[permission] {
				filtered = append(filtered, permission)
				seen[permission] = true
			}
		}
		if len(filtered) != len(s.Grants) {
			s.Grants = filtered
			if err = store.SavePluginSettings(ctx, id, s); err != nil {
				return nil, err
			}
		}
		m.entries[id] = &entry{def: d, settings: s, status: "disabled", jobs: map[string]context.CancelFunc{}}
	}
	if err := store.RecoverPluginRuns(ctx); err != nil {
		return nil, err
	}
	for _, v := range m.List() {
		if v.Settings.Enabled {
			_ = m.Enable(ctx, v.Manifest.ID)
		}
	}
	return m, nil
}
func (m *Manager) List() []View {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := []View{}
	for id, e := range m.entries {
		m.checkHealth(e)
		record, external := m.packages[id]
		out = append(out, View{Manifest: clone(e.def.Manifest), Settings: clone(e.settings), Status: e.status, Error: e.err, Active: e.active, External: external, Rollback: record.Previous != nil})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Manifest.ID < out[j].Manifest.ID })
	return out
}
func (m *Manager) get(id string) (*entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.entries[id]
	if e == nil {
		return nil, ErrNotFound
	}
	return e, nil
}
func granted(s Settings, p string) bool {
	for _, g := range s.Grants {
		if g == p {
			return true
		}
	}
	return false
}
func (m *Manager) Configure(ctx context.Context, id string, c json.RawMessage, grants []string) error {
	e, err := m.get(id)
	if err != nil {
		return err
	}
	e.admin.Lock()
	defer e.admin.Unlock()
	m.mu.Lock()
	s := clone(e.settings)
	busy := e.status != "disabled" && e.status != "failed"
	closed := m.closed
	m.mu.Unlock()
	if closed || busy {
		return ErrBusy
	}
	if !object(c) {
		return fmt.Errorf("config must be a JSON object <= %d bytes", MaxPayload)
	}
	known := map[string]bool{}
	for _, p := range e.def.Manifest.Permissions {
		known[p] = true
	}
	seen := map[string]bool{}
	for _, p := range grants {
		if !known[p] || seen[p] {
			return ErrPermission
		}
		seen[p] = true
	}
	if e.def.ValidateConfig != nil {
		if err = protect(func() error { return e.def.ValidateConfig(c) }); err != nil {
			return err
		}
	}
	s.Config = append(json.RawMessage(nil), c...)
	s.Grants = append([]string{}, grants...)
	if err = m.store.SavePluginSettings(ctx, id, s); err != nil {
		return err
	}
	m.mu.Lock()
	e.settings = s
	m.mu.Unlock()
	return nil
}
func (m *Manager) Enable(ctx context.Context, id string) error {
	e, err := m.get(id)
	if err != nil {
		return err
	}
	e.admin.Lock()
	defer e.admin.Unlock()
	m.mu.Lock()
	if m.closed || e.status == "removed" {
		m.mu.Unlock()
		return ErrDisabled
	}
	m.checkHealth(e)
	if e.status == "active" {
		m.mu.Unlock()
		return nil
	}
	if e.active != 0 || e.status == "starting" || e.status == "stopping" {
		m.mu.Unlock()
		return ErrBusy
	}
	s := clone(e.settings)
	if s.DataVersion != e.def.Manifest.DataVersion {
		e.status = "failed"
		e.err = "data migration required"
		m.mu.Unlock()
		return ErrConflict
	}
	for _, p := range e.def.Manifest.Permissions {
		if !granted(s, p) {
			e.status = "failed"
			e.err = ErrPermission.Error()
			m.mu.Unlock()
			return ErrPermission
		}
	}
	e.status = "starting"
	e.err = ""
	e.ctx, e.cancel = context.WithCancel(context.Background())
	e.ready = make(chan struct{})
	h := &host{manager: m, id: id, settings: s, life: e.ctx, ready: e.ready}
	m.mu.Unlock()
	startCtx, startCancel := context.WithTimeout(ctx, 10*time.Second)
	canceled := make(chan struct{})
	stopRequest := context.AfterFunc(startCtx, func() { e.cancel(); close(canceled) })
	err = protect(func() error {
		if e.ctx.Err() != nil {
			return e.ctx.Err()
		}
		if e.def.ValidateConfig != nil {
			if er := e.def.ValidateConfig(s.Config); er != nil {
				return er
			}
		}
		if e.def.Start != nil {
			return e.def.Start(e.ctx, h)
		}
		return nil
	})
	if !stopRequest() {
		<-canceled
	}
	if err == nil && startCtx.Err() != nil {
		err = startCtx.Err()
	}
	startCancel()
	if err == nil && e.ctx.Err() != nil {
		err = e.ctx.Err()
	}
	if err == nil {
		s.Enabled = true
		err = m.store.SavePluginSettings(ctx, id, s)
	}
	if err != nil {
		e.cancel()
		if e.def.Stop != nil {
			stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = protect(func() error { return e.def.Stop(stopCtx) })
			cancel()
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		e.status = "failed"
		e.err = err.Error()
		return err
	}
	e.settings = s
	e.status = "active"
	close(e.ready)
	return nil
}
func (m *Manager) Disable(ctx context.Context, id string) error { return m.disable(ctx, id, true) }
func (m *Manager) disable(ctx context.Context, id string, persist bool) error {
	e, err := m.get(id)
	if err != nil {
		return err
	}
	e.admin.Lock()
	defer e.admin.Unlock()
	m.mu.Lock()
	s := clone(e.settings)
	removed := e.status == "removed"
	m.mu.Unlock()
	if removed {
		return ErrNotFound
	}
	if persist {
		s.Enabled = false
		if err = m.store.SavePluginSettings(ctx, id, s); err != nil {
			return err
		}
	}
	m.mu.Lock()
	e.settings = s
	if e.status == "disabled" || e.status == "failed" {
		m.mu.Unlock()
		return nil
	}
	e.status = "stopping"
	if e.cancel != nil {
		e.cancel()
	}
	idle := e.idle
	active := e.active
	m.mu.Unlock()
	if active > 0 {
		select {
		case <-idle:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if e.def.Stop != nil {
		err = protect(func() error { return e.def.Stop(ctx) })
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		e.err = err.Error()
		e.status = "failed"
	} else {
		e.status = "disabled"
		e.err = ""
	}
	return err
}
func (m *Manager) Close(ctx context.Context) error {
	m.catalog.Lock()
	defer m.catalog.Unlock()
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
	var first error
	for _, v := range m.List() {
		if err := m.disable(ctx, v.Manifest.ID, false); err != nil && first == nil {
			first = err
		}
	}
	return first
}
func (m *Manager) Migrate(ctx context.Context, id string) error {
	e, err := m.get(id)
	if err != nil {
		return err
	}
	e.admin.Lock()
	defer e.admin.Unlock()
	m.mu.Lock()
	s := clone(e.settings)
	busy := e.status == "removed" || e.active > 0 || e.status == "active" || e.status == "starting" || e.status == "stopping"
	m.mu.Unlock()
	if busy || s.Enabled {
		return ErrBusy
	}
	if s.DataVersion == e.def.Manifest.DataVersion {
		return nil
	}
	if e.def.Migrate == nil || s.DataVersion > e.def.Manifest.DataVersion {
		return ErrConflict
	}
	err = m.store.MigratePlugin(ctx, id, s.DataVersion, e.def.Manifest.DataVersion, func(values map[string]json.RawMessage) error {
		return protect(func() error { return e.def.Migrate(ctx, s.DataVersion, e.def.Manifest.DataVersion, values) })
	})
	if err == nil {
		m.mu.Lock()
		e.settings.DataVersion = e.def.Manifest.DataVersion
		e.status = "disabled"
		e.err = ""
		m.mu.Unlock()
	}
	return err
}
func (m *Manager) acquire(id, op string) (*entry, Operation, *host, func(), error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.entries[id]
	if e == nil {
		return nil, Operation{}, nil, nil, ErrNotFound
	}
	m.checkHealth(e)
	if m.closed || e.status != "active" {
		return nil, Operation{}, nil, nil, ErrDisabled
	}
	var operation Operation
	found := false
	for _, o := range e.def.Manifest.Operations {
		if o.Name == op {
			operation = o
			found = true
			break
		}
	}
	if !found {
		return nil, operation, nil, nil, ErrNotFound
	}
	if e.active > 0 {
		return nil, operation, nil, nil, ErrBusy
	}
	e.active++
	e.idle = make(chan struct{})
	h := &host{manager: m, id: id, settings: clone(e.settings), life: e.ctx, ready: e.ready}
	done := func() {
		m.mu.Lock()
		e.active--
		if e.active == 0 {
			close(e.idle)
		}
		m.mu.Unlock()
	}
	return e, operation, h, done, nil
}
func execute(ctx context.Context, e *entry, o Operation, h *host, r Request) (result json.RawMessage, err error) {
	if !object(r.Input) {
		return nil, fmt.Errorf("invalid or oversized JSON input")
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(o.TimeoutSeconds)*time.Second)
	defer cancel()
	stop := context.AfterFunc(h.life, cancel)
	defer stop()
	if h.life.Err() != nil {
		return nil, ErrDisabled
	}
	err = protect(func() error { var er error; result, er = e.def.Handle(ctx, h, o.Name, clone(r)); return er })
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err == nil && !validJSON(result) {
		err = fmt.Errorf("plugin returned invalid or oversized JSON")
	}
	if err != nil {
		return nil, err
	}
	return result, err
}
func (m *Manager) Call(ctx context.Context, id, op string, r Request) (json.RawMessage, error) {
	e, o, h, done, err := m.acquire(id, op)
	if err != nil {
		return nil, err
	}
	defer done()
	if !object(r.Input) {
		return nil, fmt.Errorf("invalid or oversized JSON input")
	}
	b := make([]byte, 16)
	if _, err = rand.Read(b); err != nil {
		return nil, err
	}
	idRun := hex.EncodeToString(b)
	run := Run{ID: idRun, PluginID: id, Key: idRun, Operation: op, Request: clone(r), Status: "running", CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
	if _, _, err = m.store.CreatePluginRun(ctx, run); err != nil {
		return nil, err
	}
	callCtx, callCancel := context.WithCancel(ctx)
	defer callCancel()
	m.mu.Lock()
	e.jobs[idRun] = callCancel
	m.mu.Unlock()
	defer func() { m.mu.Lock(); delete(e.jobs, idRun); m.mu.Unlock() }()
	result, err := execute(callCtx, e, o, h, r)
	run.Status = "completed"
	run.Result = result
	run.UpdatedAt = time.Now().UTC()
	if err != nil {
		run.Status = "failed"
		run.Error = err.Error()
		if callCtx.Err() != nil || h.life.Err() != nil {
			run.Status = "canceled"
		} else if err == context.DeadlineExceeded {
			run.Status = "timed_out"
		}
	}
	saveCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if saveErr := m.store.FinishPluginRun(saveCtx, run); saveErr != nil {
		return nil, fmt.Errorf("persist plugin result: %w", saveErr)
	}
	return result, err
}
func (m *Manager) Tools() []View {
	out := []View{}
	for _, v := range m.List() {
		if v.Status == "active" && granted(v.Settings, "tools") {
			out = append(out, v)
		}
	}
	return out
}
func (m *Manager) Submit(ctx context.Context, id, op, key string, r Request) (Run, error) {
	if !object(r.Input) || len(key) > 128 {
		return Run{}, fmt.Errorf("invalid job request")
	}
	if key != "" {
		entry, err := m.get(id)
		if err != nil {
			return Run{}, err
		}
		m.mu.Lock()
		allowed := entry.status == "active" && !m.closed && granted(entry.settings, "jobs")
		m.mu.Unlock()
		if !allowed {
			return Run{}, ErrDisabled
		}
		old, err := m.store.PluginRunByKey(ctx, id, key)
		if err == nil {
			if old.Operation != op || old.Request.SessionID != r.SessionID || string(old.Request.Input) != string(r.Input) {
				return Run{}, ErrConflict
			}
			return old, nil
		}
		if !errors.Is(err, ErrNotFound) {
			return Run{}, err
		}
	}
	e, o, h, done, err := m.acquire(id, op)
	if err != nil {
		return Run{}, err
	}
	if !o.Background || !granted(h.settings, "jobs") {
		done()
		return Run{}, ErrPermission
	}
	b := make([]byte, 16)
	if _, err = rand.Read(b); err != nil {
		done()
		return Run{}, err
	}
	rid := hex.EncodeToString(b)
	if key == "" {
		key = rid
	}
	run := Run{ID: rid, PluginID: id, Key: key, Operation: op, Request: clone(r), Status: "running", CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
	saved, created, err := m.store.CreatePluginRun(ctx, run)
	if err != nil || !created {
		done()
		return saved, err
	}
	jobCtx, cancel := context.WithCancel(h.life)
	m.mu.Lock()
	e.jobs[rid] = cancel
	m.mu.Unlock()
	go func() {
		defer done()
		defer cancel()
		defer func() { m.mu.Lock(); delete(e.jobs, rid); m.mu.Unlock() }()
		result, er := execute(jobCtx, e, o, h, r)
		run.Result = result
		run.Status = "completed"
		if er != nil {
			run.Status = "failed"
			run.Error = er.Error()
			if jobCtx.Err() != nil {
				run.Status = "canceled"
			} else if er == context.DeadlineExceeded {
				run.Status = "timed_out"
			}
		}
		run.UpdatedAt = time.Now().UTC()
		saveCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		if er = m.store.FinishPluginRun(saveCtx, run); er != nil {
			m.mu.Lock()
			e.err = "persist job result: " + er.Error()
			m.mu.Unlock()
		}
	}()
	return saved, nil
}
func (m *Manager) Cancel(id, runID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.entries[id]
	if e == nil {
		return ErrNotFound
	}
	c := e.jobs[runID]
	if c == nil {
		return ErrNotFound
	}
	c()
	return nil
}
func (m *Manager) Runs(ctx context.Context, id string) ([]Run, error) {
	if _, err := m.get(id); err != nil {
		return nil, err
	}
	return m.store.PluginRuns(ctx, id)
}

type host struct {
	ready    <-chan struct{}
	manager  *Manager
	id       string
	settings Settings
	life     context.Context
}

func (h *host) Ready() <-chan struct{}  { return h.ready }
func (h *host) Config() json.RawMessage { return append(json.RawMessage(nil), h.settings.Config...) }
func (h *host) check(ctx context.Context, p string) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if h.life.Err() != nil {
		return ErrDisabled
	}
	if !granted(h.settings, p) {
		return ErrPermission
	}
	return nil
}
func (h *host) scope(ctx context.Context, p string) (context.Context, context.CancelFunc, error) {
	if err := h.check(ctx, p); err != nil {
		return nil, nil, err
	}
	child, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(h.life, cancel)
	return child, func() { stop(); cancel() }, nil
}
func (h *host) Get(ctx context.Context, key string) (json.RawMessage, error) {
	ctx, cancel, err := h.scope(ctx, "storage")
	if err != nil {
		return nil, err
	}
	defer cancel()
	return h.manager.store.PluginValue(ctx, h.id, key)
}
func (h *host) Put(ctx context.Context, key string, v json.RawMessage) error {
	ctx, cancel, err := h.scope(ctx, "storage")
	if err != nil {
		return err
	}
	defer cancel()
	return h.manager.store.PutPluginValue(ctx, h.id, key, v)
}
func (h *host) Delete(ctx context.Context, key string) error {
	ctx, cancel, err := h.scope(ctx, "storage")
	if err != nil {
		return err
	}
	defer cancel()
	return h.manager.store.DeletePluginValue(ctx, h.id, key)
}
func (h *host) Call(ctx context.Context, service string, r Request) (json.RawMessage, error) {
	ctx, cancel, err := h.scope(ctx, service)
	if err != nil {
		return nil, err
	}
	defer cancel()
	fn := h.manager.services[service]
	if fn == nil {
		return nil, ErrNotFound
	}
	if !object(r.Input) {
		return nil, fmt.Errorf("invalid service input")
	}
	var out json.RawMessage
	err = protect(func() error { var er error; out, er = fn(ctx, clone(r)); return er })
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err == nil && !validJSON(out) {
		return nil, fmt.Errorf("invalid service result")
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}
func (h *host) Submit(ctx context.Context, op, key string, r Request) (Run, error) {
	ctx, cancel, err := h.scope(ctx, "jobs")
	if err != nil {
		return Run{}, err
	}
	defer cancel()
	return h.manager.Submit(ctx, h.id, op, key, r)
}

func (h *host) Runs(ctx context.Context) ([]Run, error) {
	ctx, cancel, err := h.scope(ctx, "jobs")
	if err != nil {
		return nil, err
	}
	defer cancel()
	return h.manager.Runs(ctx, h.id)
}
func (h *host) Cancel(ctx context.Context, runID string) error {
	if err := h.check(ctx, "jobs"); err != nil {
		return err
	}
	return h.manager.Cancel(h.id, runID)
}

// Called with m.mu held. A dead external process revokes all old Host handles.
func (m *Manager) checkHealth(e *entry) {
	if e.status == "active" && e.def.Health != nil {
		if err := e.def.Health(); err != nil {
			e.status = "failed"
			e.err = err.Error()
			if e.cancel != nil {
				e.cancel()
			}
		}
	}
}
