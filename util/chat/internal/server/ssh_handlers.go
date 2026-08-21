package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/extra"
)

var sshIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

func (s *Server) sshHosts(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		hosts, err := s.db.SSHHosts()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, hosts)
	case http.MethodPost:
		host, err := decodeSSHHost(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		host.ID = fmt.Sprintf("ssh-%d", time.Now().UnixNano())
		host, err = s.db.CreateSSHHost(host)
		if err != nil {
			http.Error(w, friendlySSHDBError(err), http.StatusConflict)
			return
		}
		writeJSON(w, http.StatusCreated, host)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) sshHost(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/ssh/hosts/"), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	host, err := s.db.SSHHost(parts[0])
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if len(parts) == 1 {
		switch r.Method {
		case http.MethodPatch:
			next, err := decodeSSHHost(w, r)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			next.ID = host.ID
			next, err = s.db.UpdateSSHHost(next)
			if err != nil {
				http.Error(w, friendlySSHDBError(err), http.StatusConflict)
				return
			}
			if err := s.db.DeleteSSHHostGrants(host.ID); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, http.StatusOK, next)
		case http.MethodDelete:
			if err := s.db.DeleteSSHHost(host.ID); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			methodNotAllowed(w)
		}
		return
	}
	if len(parts) != 2 || r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}
	target := extra.Target{Host: host.Hostname, Port: host.Port, User: host.Username, KeyID: host.KeyID}
	switch parts[1] {
	case "test":
		if err := s.extraSnapshot().Check(r.Context(), target); err != nil {
			writeExtraError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
	case "trust":
		var request struct {
			PublicKey string `json:"public_key"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32*1024)).Decode(&request); err != nil || strings.TrimSpace(request.PublicKey) == "" {
			http.Error(w, "public_key is required", http.StatusBadRequest)
			return
		}
		result, err := s.extraSnapshot().Trust(r.Context(), host.Hostname, host.Port, request.PublicKey)
		if err != nil {
			writeExtraError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, result)
	default:
		http.NotFound(w, r)
	}
}

func decodeSSHHost(w http.ResponseWriter, r *http.Request) (db.SSHHost, error) {
	var host db.SSHHost
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32*1024))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&host); err != nil {
		return db.SSHHost{}, errors.New("invalid SSH server settings")
	}
	host.Alias = strings.TrimSpace(host.Alias)
	host.Name = strings.TrimSpace(host.Name)
	host.Hostname = strings.TrimSpace(host.Hostname)
	host.Username = strings.TrimSpace(host.Username)
	host.KeyID = strings.TrimSpace(host.KeyID)
	if !sshIdentifierPattern.MatchString(host.Alias) {
		return db.SSHHost{}, errors.New("alias may contain only letters, numbers, period, underscore, and hyphen")
	}
	if host.Name == "" || len(host.Name) > 100 {
		return db.SSHHost{}, errors.New("display name is required and must be at most 100 characters")
	}
	if host.Hostname == "" || len(host.Hostname) > 253 || strings.ContainsAny(host.Hostname, " /\\\t\r\n") {
		return db.SSHHost{}, errors.New("invalid SSH hostname")
	}
	if host.Port == 0 {
		host.Port = 22
	}
	if host.Port < 1 || host.Port > 65535 {
		return db.SSHHost{}, errors.New("invalid SSH port")
	}
	if host.Username == "" || len(host.Username) > 128 {
		return db.SSHHost{}, errors.New("SSH username is required")
	}
	if !sshIdentifierPattern.MatchString(host.KeyID) {
		return db.SSHHost{}, errors.New("invalid SSH key id")
	}
	if host.TimeoutSeconds == 0 {
		host.TimeoutSeconds = 60
	}
	if host.TimeoutSeconds < 1 || host.TimeoutSeconds > 86400 {
		return db.SSHHost{}, errors.New("timeout must be between 1 and 86400 seconds")
	}
	return host, nil
}

func writeExtraError(w http.ResponseWriter, err error) {
	var apiErr *extra.HTTPError
	if errors.As(err, &apiErr) {
		payload := map[string]any{"error": apiErr.Message}
		if apiErr.HostKey != nil {
			payload["host_key"] = apiErr.HostKey
		}
		writeJSON(w, apiErr.Status, payload)
		return
	}
	writeJSON(w, http.StatusBadGateway, map[string]any{"error": err.Error()})
}

func friendlySSHDBError(err error) string {
	if strings.Contains(strings.ToLower(err.Error()), "unique") {
		return "SSH server alias is already in use"
	}
	return err.Error()
}
