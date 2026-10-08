// Package plugins manages built-in extensions and sandboxed external packages.
package plugins

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

const APIVersion = 1
const MaxPayload = 256 << 10

var (
	ErrNotFound   = errors.New("plugin or operation not found")
	ErrDisabled   = errors.New("plugin is not active")
	ErrPermission = errors.New("plugin permission not granted")
	ErrBusy       = errors.New("plugin is busy")
	ErrConflict   = errors.New("plugin state conflict")
)

type Operation struct {
	Name           string          `json:"name"`
	Description    string          `json:"description"`
	Parameters     json.RawMessage `json:"parameters"`
	Tool           bool            `json:"tool"`
	Background     bool            `json:"background"`
	TimeoutSeconds int             `json:"timeout_seconds"`
}
type Panel struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Operations  []string `json:"operations"`
}
type Manifest struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Version     string      `json:"version"`
	API         int         `json:"api_version"`
	DataVersion int         `json:"data_version"`
	Description string      `json:"description"`
	Permissions []string    `json:"permissions"`
	Operations  []Operation `json:"operations"`
	Panels      []Panel     `json:"panels"`
}
type Settings struct {
	Enabled     bool            `json:"enabled"`
	Config      json.RawMessage `json:"config"`
	Grants      []string        `json:"grants"`
	DataVersion int             `json:"data_version"`
}
type View struct {
	Manifest Manifest `json:"manifest"`
	Settings Settings `json:"settings"`
	Status   string   `json:"status"`
	Error    string   `json:"error,omitempty"`
	Active   int      `json:"active_runs"`
	External bool     `json:"external"`
	Rollback bool     `json:"rollback_available"`
}
type Request struct {
	SessionID string          `json:"session_id,omitempty"`
	Input     json.RawMessage `json:"input"`
}
type Run struct {
	ID        string          `json:"id"`
	PluginID  string          `json:"plugin_id"`
	Key       string          `json:"key"`
	Operation string          `json:"operation"`
	Request   Request         `json:"request"`
	Status    string          `json:"status"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
}
type Host interface {
	// Ready closes only after activation and durable enablement succeed.
	Ready() <-chan struct{}
	Config() json.RawMessage
	Get(context.Context, string) (json.RawMessage, error)
	Put(context.Context, string, json.RawMessage) error
	Delete(context.Context, string) error
	Call(context.Context, string, Request) (json.RawMessage, error)
	Submit(context.Context, string, string, Request) (Run, error)
	Runs(context.Context) ([]Run, error)
	Cancel(context.Context, string) error
}
type Definition struct {
	Health         func() error
	Manifest       Manifest
	ValidateConfig func(json.RawMessage) error
	Start          func(context.Context, Host) error
	Stop           func(context.Context) error
	Handle         func(context.Context, Host, string, Request) (json.RawMessage, error)
	// Migrate edits a detached namespace. The store commits it only on success.
	Migrate func(context.Context, int, int, map[string]json.RawMessage) error
}
type Service func(context.Context, Request) (json.RawMessage, error)
type Store interface {
	PluginSettings(context.Context, string) (Settings, error)
	SavePluginSettings(context.Context, string, Settings) error
	PluginValue(context.Context, string, string) (json.RawMessage, error)
	PutPluginValue(context.Context, string, string, json.RawMessage) error
	DeletePluginValue(context.Context, string, string) error
	MigratePlugin(context.Context, string, int, int, func(map[string]json.RawMessage) error) error
	CreatePluginRun(context.Context, Run) (Run, bool, error)
	PluginRunByKey(context.Context, string, string) (Run, error)
	FinishPluginRun(context.Context, Run) error
	PluginRuns(context.Context, string) ([]Run, error)
	RecoverPluginRuns(context.Context) error
}

// Builtins is the discovery/catalog entry point. Business plugins are registered
// here in a separate development phase; test fixtures never enter this catalog.
func Builtins() []Definition { return nil }
