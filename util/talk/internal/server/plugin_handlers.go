package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	"sparktalk/internal/llm"
	"sparktalk/internal/plugins"
)

func (s *Server) registerPluginTools(registry *completionToolRegistry) {
	if s.plugins == nil {
		return
	}
	for _, v := range s.plugins.Tools() {
		for _, op := range v.Manifest.Operations {
			if !op.Tool {
				continue
			}
			id, name, background := v.Manifest.ID, op.Name, op.Background
			registry.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: plugins.ToolName(id, name), Description: op.Description, Parameters: op.Parameters}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
				request := plugins.Request{SessionID: registry.sessionID, Input: json.RawMessage(call.Function.Arguments)}
				if background {
					key := ""
					if call.ID != "" {
						key = fmt.Sprintf("tool_%x", sha256.Sum256([]byte(registry.sessionID+"\x00"+call.ID)))
					}
					run, err := s.plugins.Submit(ctx, id, name, key, request)
					if err != nil {
						return registeredToolResult{}, err
					}
					raw, err := json.Marshal(run)
					return registeredToolResult{Result: string(raw)}, err
				}
				result, err := s.plugins.Call(ctx, id, name, request)
				return registeredToolResult{Result: string(result)}, err
			})
		}
	}
}
func pluginError(w http.ResponseWriter, err error) {
	status := http.StatusBadRequest
	switch {
	case errors.Is(err, plugins.ErrNotFound):
		status = 404
	case errors.Is(err, plugins.ErrPermission):
		status = 403
	case errors.Is(err, plugins.ErrBusy), errors.Is(err, plugins.ErrDisabled), errors.Is(err, plugins.ErrConflict):
		status = 409
	case errors.Is(err, context.DeadlineExceeded):
		status = 504
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
func decodePluginBody(w http.ResponseWriter, r *http.Request, v any) error {
	media, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || media != "application/json" {
		return fmt.Errorf("Content-Type must be application/json")
	}
	if origin := r.Header.Get("Origin"); origin != "" {
		u, err := url.Parse(origin)
		if err != nil || u.Host != r.Host {
			return plugins.ErrPermission
		}
	}
	r.Body = http.MaxBytesReader(w, r.Body, plugins.MaxPayload+4096)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if dec.Decode(&struct{}{}) != io.EOF {
		return fmt.Errorf("unexpected trailing JSON")
	}
	return nil
}
func (s *Server) pluginAPI(w http.ResponseWriter, r *http.Request) {
	if s.plugins == nil {
		writeJSON(w, 503, map[string]string{"error": "plugin runtime unavailable"})
		return
	}
	if r.URL.Path == "/api/plugins/install" {
		if r.Method != "POST" {
			methodNotAllowed(w)
			return
		}
		media, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || media != "application/zip" {
			pluginError(w, fmt.Errorf("Content-Type must be application/zip"))
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" {
			u, er := url.Parse(origin)
			if er != nil || u.Host != r.Host {
				pluginError(w, plugins.ErrPermission)
				return
			}
		}
		r.Body = http.MaxBytesReader(w, r.Body, plugins.MaxPackage)
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		if err = s.plugins.InstallPackage(ctx, r.Body); err != nil {
			pluginError(w, err)
			return
		}
		writeJSON(w, 201, map[string]bool{"ok": true})
		return
	}
	if r.URL.Path == "/api/plugins" {
		if r.Method != "GET" {
			methodNotAllowed(w)
			return
		}
		writeJSON(w, 200, s.plugins.List())
		return
	}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/plugins/"), "/")
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}
	id, action := parts[0], parts[1]
	if action == "runs" && r.Method == "GET" {
		runs, err := s.plugins.Runs(r.Context(), id)
		if err != nil {
			pluginError(w, err)
			return
		}
		writeJSON(w, 200, runs)
		return
	}
	if r.Method != "POST" && !(action == "configure" && r.Method == "PUT") {
		methodNotAllowed(w)
		return
	}
	var in struct {
		Config    json.RawMessage `json:"config"`
		Grants    []string        `json:"grants"`
		Operation string          `json:"operation"`
		Key       string          `json:"key"`
		Request   plugins.Request `json:"request"`
		Purge     bool            `json:"purge"`
		RunID     string          `json:"run_id"`
	}
	if err := decodePluginBody(w, r, &in); err != nil {
		pluginError(w, err)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	var err error
	switch action {
	case "remove":
		err = s.plugins.RemovePackage(ctx, id, in.Purge)
	case "rollback":
		err = s.plugins.RollbackPackage(ctx, id)
	case "configure":
		err = s.plugins.Configure(ctx, id, in.Config, in.Grants)
	case "enable":
		err = s.plugins.Enable(ctx, id)
	case "disable":
		err = s.plugins.Disable(ctx, id)
	case "migrate":
		err = s.plugins.Migrate(ctx, id)
	case "call":
		var result json.RawMessage
		result, err = s.plugins.Call(ctx, id, in.Operation, in.Request)
		if err == nil {
			writeJSON(w, 200, result)
			return
		}
	case "submit":
		var run plugins.Run
		run, err = s.plugins.Submit(ctx, id, in.Operation, in.Key, in.Request)
		if err == nil {
			writeJSON(w, 202, run)
			return
		}
	case "cancel":
		err = s.plugins.Cancel(id, in.RunID)
	default:
		http.NotFound(w, r)
		return
	}
	if err != nil {
		pluginError(w, err)
		return
	}
	writeJSON(w, 200, map[string]bool{"ok": true})
}
