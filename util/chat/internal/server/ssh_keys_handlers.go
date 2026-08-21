package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

const maxSSHPrivateKeyBytes = 128 * 1024

func (s *Server) sshKeys(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		keys, err := s.extraSnapshot().Keys(r.Context())
		if err != nil {
			writeExtraError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, keys)
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxSSHPrivateKeyBytes+64*1024)
	if err := r.ParseMultipartForm(maxSSHPrivateKeyBytes + 64*1024); err != nil {
		http.Error(w, "invalid or oversized SSH key upload", http.StatusBadRequest)
		return
	}
	keyID := strings.TrimSpace(r.FormValue("key_id"))
	file, _, err := r.FormFile("key")
	if err != nil {
		http.Error(w, "SSH private key file is required", http.StatusBadRequest)
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxSSHPrivateKeyBytes+1))
	if err != nil || len(data) == 0 || len(data) > maxSSHPrivateKeyBytes {
		http.Error(w, "SSH private key must be between 1 byte and 128 KiB", http.StatusBadRequest)
		return
	}
	key, err := s.extraSnapshot().ImportKey(r.Context(), keyID, data)
	if err != nil {
		writeExtraError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, key)
}

func (s *Server) sshKey(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/ssh/keys/"), "/")
	if rest == "generate" {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		var request struct {
			KeyID string `json:"key_id"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			http.Error(w, "invalid SSH key request", http.StatusBadRequest)
			return
		}
		key, err := s.extraSnapshot().GenerateKey(r.Context(), strings.TrimSpace(request.KeyID))
		if err != nil {
			writeExtraError(w, err)
			return
		}
		writeJSON(w, http.StatusCreated, key)
		return
	}
	if rest == "" || strings.Contains(rest, "/") {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodDelete {
		methodNotAllowed(w)
		return
	}
	hosts, err := s.db.SSHHosts()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, host := range hosts {
		if host.KeyID == rest {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "SSH key is used by server profile " + host.Name})
			return
		}
	}
	if err := s.extraSnapshot().DeleteKey(r.Context(), rest); err != nil {
		writeExtraError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
