package plugins

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const MaxPackage = 128 << 20

type PackageManifest struct {
	Manifest
	Executable string `json:"executable"`
	Platform   string `json:"platform"`
}
type PackageRecord struct {
	Manifest      PackageManifest `json:"manifest"`
	ExecutableSHA string          `json:"executable_sha256"`
	Digest        string          `json:"digest"`
	Previous      *PackageBackup  `json:"previous,omitempty"`
}
type PackageBackup struct {
	Package  PackageRecord              `json:"package"`
	Settings Settings                   `json:"settings"`
	Values   map[string]json.RawMessage `json:"values"`
}
type PackageStore interface {
	PluginPackages(context.Context) ([]PackageRecord, error)
	PluginSnapshot(context.Context, string) (Settings, map[string]json.RawMessage, error)
	// Atomically commits catalog, settings and the entire private namespace.
	CommitPluginPackage(context.Context, string, *PackageRecord, Settings, map[string]json.RawMessage, bool) error
}

func (m *Manager) OpenPackages(ctx context.Context, root string) error {
	m.catalog.Lock()
	defer m.catalog.Unlock()
	root, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(root, 0700); err != nil {
		return err
	}
	store, ok := m.store.(PackageStore)
	if !ok {
		return fmt.Errorf("package storage unavailable")
	}
	records, err := store.PluginPackages(ctx)
	if err != nil {
		return err
	}
	m.packageRoot = root
	for _, record := range records {
		if err = m.validatePackage(record.Manifest); err != nil {
			return err
		}
		if len(record.Digest) != 64 {
			return fmt.Errorf("invalid stored package digest")
		}
		if _, err = hex.DecodeString(record.Digest); err != nil {
			return err
		}
		id := record.Manifest.ID
		if _, err = m.get(id); err == nil {
			return fmt.Errorf("package ID conflicts with builtin")
		}
		settings, err := m.store.PluginSettings(ctx, id)
		if err != nil {
			return err
		}
		e := &entry{def: externalDefinition(record.Manifest.Manifest, m.executable(record), record.ExecutableSHA), settings: settings, status: "disabled", jobs: map[string]context.CancelFunc{}}
		m.mu.Lock()
		m.entries[id] = e
		m.packages[id] = record
		m.mu.Unlock()
		if settings.Enabled {
			_ = m.Enable(ctx, id)
		}
	}
	return nil
}
func (m *Manager) executable(r PackageRecord) string {
	return filepath.Join(m.packageRoot, r.Manifest.ID, r.Digest, "plugin")
}
func (m *Manager) validatePackage(p PackageManifest) error {
	if p.Executable != "plugin" || p.Platform != runtime.GOOS+"/"+runtime.GOARCH {
		return fmt.Errorf("package requires executable=plugin and platform=%s/%s", runtime.GOOS, runtime.GOARCH)
	}
	return validate(Definition{Manifest: p.Manifest, Handle: func(context.Context, Host, string, Request) (json.RawMessage, error) { return nil, nil }}, m.services)
}
func (m *Manager) stage(reader io.Reader) (PackageRecord, string, error) {
	var record PackageRecord
	if m.packageRoot == "" {
		return record, "", fmt.Errorf("external package storage unavailable")
	}
	dir, err := os.MkdirTemp(m.packageRoot, ".stage-")
	if err != nil {
		return record, "", err
	}
	success := false
	defer func() {
		if !success {
			os.RemoveAll(dir)
		}
	}()
	file, err := os.Create(filepath.Join(dir, "upload.zip"))
	if err != nil {
		return record, "", err
	}
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(file, hash), io.LimitReader(reader, MaxPackage+1))
	file.Close()
	if err != nil {
		return record, "", err
	}
	if n > MaxPackage {
		return record, "", fmt.Errorf("package exceeds 128 MiB")
	}
	archive, err := zip.OpenReader(filepath.Join(dir, "upload.zip"))
	if err != nil {
		return record, "", err
	}
	defer archive.Close()
	if len(archive.File) != 2 {
		return record, "", fmt.Errorf("package must contain only plugin.json and plugin")
	}
	seen := map[string]bool{}
	for _, f := range archive.File {
		if (f.Name != "plugin.json" && f.Name != "plugin") || !f.Mode().IsRegular() || seen[f.Name] {
			return record, "", fmt.Errorf("invalid package entry: %s", f.Name)
		}
		seen[f.Name] = true
		limit := int64(MaxPackage)
		if f.Name == "plugin.json" {
			limit = MaxPayload
		}
		if f.UncompressedSize64 > uint64(limit) {
			return record, "", fmt.Errorf("expanded package too large")
		}
		input, err := f.Open()
		if err != nil {
			return record, "", err
		}
		output, err := os.OpenFile(filepath.Join(dir, f.Name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			input.Close()
			return record, "", err
		}
		n, err := io.Copy(output, io.LimitReader(input, limit+1))
		input.Close()
		syncErr := output.Sync()
		closeErr := output.Close()
		if err != nil {
			return record, "", err
		}
		if syncErr != nil {
			return record, "", syncErr
		}
		if closeErr != nil {
			return record, "", closeErr
		}
		if n > limit {
			return record, "", fmt.Errorf("expanded package too large")
		}
	}
	raw, err := os.ReadFile(filepath.Join(dir, "plugin.json"))
	if err != nil {
		return record, "", err
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&record.Manifest); err != nil {
		return record, "", err
	}
	if dec.Decode(new(any)) != io.EOF {
		return record, "", fmt.Errorf("trailing manifest data")
	}
	if err = m.validatePackage(record.Manifest); err != nil {
		return record, "", err
	}
	binary := filepath.Join(dir, "plugin")
	f, err := elf.Open(binary)
	if err != nil {
		return record, "", fmt.Errorf("plugin must be a static Linux ELF executable: %w", err)
	}
	defer f.Close()
	for _, p := range f.Progs {
		if p.Type == elf.PT_INTERP {
			return record, "", fmt.Errorf("dynamic executables unsupported; build Go plugins with CGO_ENABLED=0")
		}
	}
	machine := elf.EM_AARCH64
	if runtime.GOARCH == "amd64" {
		machine = elf.EM_X86_64
	}
	if f.Machine != machine {
		return record, "", fmt.Errorf("executable architecture mismatch")
	}
	if err = os.Chmod(binary, 0500); err != nil {
		return record, "", err
	}
	record.ExecutableSHA, err = fileDigest(binary)
	if err != nil {
		return record, "", err
	}
	record.Digest = hex.EncodeToString(hash.Sum(nil))
	os.Remove(filepath.Join(dir, "upload.zip"))
	success = true
	return record, dir, nil
}

// probeHost allows no application side effects during installation/migration.
type probeHost struct {
	config json.RawMessage
	ready  chan struct{}
}

func (h probeHost) Ready() <-chan struct{}                               { return h.ready }
func (h probeHost) Config() json.RawMessage                              { return h.config }
func (h probeHost) Get(context.Context, string) (json.RawMessage, error) { return nil, ErrPermission }
func (h probeHost) Put(context.Context, string, json.RawMessage) error   { return ErrPermission }
func (h probeHost) Delete(context.Context, string) error                 { return ErrPermission }
func (h probeHost) Call(context.Context, string, Request) (json.RawMessage, error) {
	return nil, ErrPermission
}
func (h probeHost) Submit(context.Context, string, string, Request) (Run, error) {
	return Run{}, ErrPermission
}
func (h probeHost) Runs(context.Context) ([]Run, error)  { return nil, ErrPermission }
func (h probeHost) Cancel(context.Context, string) error { return ErrPermission }
func probe(ctx context.Context, r PackageRecord, path string, s Settings, values map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	ctx, c := context.WithTimeout(ctx, 10*time.Second)
	defer c()
	ready := make(chan struct{})
	close(ready)
	p := &processPlugin{manifest: r.Manifest.Manifest, executable: path, expectedHash: r.ExecutableSHA}
	if err := p.start(ctx, probeHost{s.Config, ready}); err != nil {
		return nil, err
	}
	defer func() { stop, c := context.WithTimeout(context.Background(), 3*time.Second); defer c(); p.close(stop) }()
	if s.DataVersion != 0 && s.DataVersion != r.Manifest.DataVersion {
		if s.DataVersion > r.Manifest.DataVersion {
			return nil, fmt.Errorf("use rollback for data downgrade")
		}
		var migrated map[string]json.RawMessage
		if err := p.peer.Call(ctx, "migrate", map[string]any{"from": s.DataVersion, "to": r.Manifest.DataVersion, "values": values}, &migrated); err != nil {
			return nil, err
		}
		if migrated == nil {
			return nil, fmt.Errorf("migration must return an object")
		}
		return migrated, nil
	}
	return values, nil
}
func (m *Manager) InstallPackage(ctx context.Context, reader io.Reader) error {
	m.catalog.Lock()
	defer m.catalog.Unlock()
	r, stage, err := m.stage(reader)
	if err != nil {
		return err
	}
	defer os.RemoveAll(stage)
	id := r.Manifest.ID
	m.mu.Lock()
	e := m.entries[id]
	old, external := m.packages[id]
	closed := m.closed
	m.mu.Unlock()
	if closed {
		return ErrDisabled
	}
	if e != nil && !external {
		return fmt.Errorf("builtin ID conflict")
	}
	if e != nil {
		e.admin.Lock()
		defer e.admin.Unlock()
		m.mu.Lock()
		busy := e.active != 0 || e.settings.Enabled || (e.status != "disabled" && e.status != "failed")
		m.mu.Unlock()
		if busy {
			return ErrBusy
		}
	}
	if external && !newerVersion(r.Manifest.Version, old.Manifest.Version) {
		return fmt.Errorf("update requires a newer version; use rollback to restore the previous version")
	}
	// Check generated tool names before changing anything durable.
	for _, v := range m.List() {
		if v.Manifest.ID == id {
			continue
		}
		for _, a := range v.Manifest.Operations {
			for _, b := range r.Manifest.Operations {
				if a.Tool && b.Tool && ToolName(v.Manifest.ID, a.Name) == ToolName(id, b.Name) {
					return fmt.Errorf("tool name conflict")
				}
			}
		}
	}
	store := m.store.(PackageStore)
	settings, values, err := store.PluginSnapshot(ctx, id)
	if err != nil {
		return err
	}
	if settings.DataVersion == 0 {
		settings = Settings{Config: json.RawMessage(`{}`), Grants: []string{}, DataVersion: r.Manifest.DataVersion}
	}
	if external {
		old.Previous = nil
		r.Previous = &PackageBackup{Package: old, Settings: clone(settings), Values: clone(values)}
	}
	values, err = probe(ctx, r, filepath.Join(stage, "plugin"), settings, values)
	if err != nil {
		return err
	}
	settings.Enabled = false
	settings.Grants = []string{}
	settings.DataVersion = r.Manifest.DataVersion
	target := filepath.Dir(m.executable(r))
	if err = os.MkdirAll(filepath.Dir(target), 0700); err != nil {
		return err
	}
	if _, err = os.Stat(target); err == nil {
		if external && (target == filepath.Dir(m.executable(old)) || (old.Previous != nil && target == filepath.Dir(m.executable(old.Previous.Package)))) {
			return fmt.Errorf("use rollback for an installed version")
		}
		if err = os.RemoveAll(target); err != nil {
			return err
		}
	}
	if err = syncDirectory(stage); err != nil {
		return err
	}
	if err = os.Rename(stage, target); err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			os.RemoveAll(target)
		}
	}()
	if dir, er := os.Open(filepath.Dir(target)); er == nil {
		err = dir.Sync()
		dir.Close()
		if err != nil {
			return err
		}
	} else {
		return er
	}
	if err = syncDirectory(m.packageRoot); err != nil {
		return err
	}
	if err = store.CommitPluginPackage(ctx, id, &r, settings, values, false); err != nil {
		return err
	}
	committed = true
	defer m.collectPackages(id, &r)
	m.mu.Lock()
	defer m.mu.Unlock()
	if e == nil {
		e = &entry{jobs: map[string]context.CancelFunc{}}
		m.entries[id] = e
	}
	e.def = externalDefinition(r.Manifest.Manifest, m.executable(r), r.ExecutableSHA)
	e.settings = settings
	e.status = "disabled"
	e.err = ""
	m.packages[id] = r
	return nil
}
func (m *Manager) RemovePackage(ctx context.Context, id string, purge bool) error {
	return m.changePackage(ctx, id, false, purge)
}
func (m *Manager) RollbackPackage(ctx context.Context, id string) error {
	return m.changePackage(ctx, id, true, false)
}
func (m *Manager) changePackage(ctx context.Context, id string, rollback, purge bool) error {
	m.catalog.Lock()
	defer m.catalog.Unlock()
	e, err := m.get(id)
	if err != nil {
		return err
	}
	e.admin.Lock()
	defer e.admin.Unlock()
	m.mu.Lock()
	r, ok := m.packages[id]
	busy := e.active != 0 || e.settings.Enabled || (e.status != "disabled" && e.status != "failed")
	m.mu.Unlock()
	if !ok {
		return ErrNotFound
	}
	if busy {
		return ErrBusy
	}
	store := m.store.(PackageStore)
	settings, values, err := store.PluginSnapshot(ctx, id)
	if err != nil {
		return err
	}
	var next *PackageRecord
	if rollback {
		if r.Previous == nil {
			return fmt.Errorf("no previous version")
		}
		b := r.Previous
		next = &b.Package
		settings = b.Settings
		settings.Enabled = false
		values = b.Values
		if _, err = probe(ctx, *next, m.executable(*next), settings, values); err != nil {
			return err
		}
	}
	if err = store.CommitPluginPackage(ctx, id, next, settings, values, purge); err != nil {
		return err
	}
	defer m.collectPackages(id, next)
	m.mu.Lock()
	defer m.mu.Unlock()
	if next == nil {
		e.status = "removed"
		delete(m.entries, id)
		delete(m.packages, id)
	} else {
		e.def = externalDefinition(next.Manifest.Manifest, m.executable(*next), next.ExecutableSHA)
		e.settings = settings
		e.status = "disabled"
		e.err = ""
		m.packages[id] = *next
	}
	// Version directories remain immutable; keeping them makes crash recovery and
	// rollback independent of destructive filesystem operations.
	return nil
}

// Only the current and one rollback version are retained. Catalog commits
// precede cleanup, so interruption leaves harmless orphan versions.
func (m *Manager) collectPackages(id string, current *PackageRecord) {
	keep := map[string]bool{}
	if current != nil {
		keep[current.Digest] = true
		if current.Previous != nil {
			keep[current.Previous.Package.Digest] = true
		}
	}
	root := filepath.Join(m.packageRoot, id)
	entries, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !keep[e.Name()] && len(e.Name()) == 64 {
			_ = os.RemoveAll(filepath.Join(root, e.Name()))
		}
	}
	if current == nil {
		_ = os.Remove(root)
	}
}

func syncDirectory(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

func newerVersion(a, b string) bool {
	aa, bb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < 3; i++ {
		if len(aa[i]) != len(bb[i]) {
			return len(aa[i]) > len(bb[i])
		}
		if aa[i] != bb[i] {
			return aa[i] > bb[i]
		}
	}
	return false
}
