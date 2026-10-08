package server

import (
	"net/http"
	"strings"
)

func (s *Server) sshConversationGrants(w http.ResponseWriter, r *http.Request, sessionID string, parts []string) {
	if _, err := s.db.Session(sessionID); err != nil {
		http.NotFound(w, r)
		return
	}
	if len(parts) == 2 {
		switch r.Method {
		case http.MethodGet:
			grants, err := s.db.SSHConversationGrants(sessionID)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, http.StatusOK, grants)
		case http.MethodDelete:
			if err := s.db.DeleteSSHConversationGrants(sessionID); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			methodNotAllowed(w)
		}
		return
	}
	if len(parts) == 3 && r.Method == http.MethodDelete {
		hostID := strings.TrimSpace(parts[2])
		if hostID == "" {
			http.NotFound(w, r)
			return
		}
		if err := s.db.DeleteSSHConversationGrant(sessionID, hostID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	http.NotFound(w, r)
}
