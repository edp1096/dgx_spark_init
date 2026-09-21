package server

import (
	"net/http/httptest"
	"os"
	"sparktalk/internal/config"
	"sparktalk/internal/orchestrator"
	"strings"
	"testing"
)

func TestLocalQADPreparationAPIScope(t *testing.T) {
	runtime, err := orchestrator.NewController()
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: config.Config{Runtime: config.RuntimeConfig{DataDir: t.TempDir()}}, runtime: runtime}
	// Hold the operation lock so accepted requests stop before spawning work.
	modelPreparationMu.Lock()
	defer modelPreparationMu.Unlock()
	for _, tc := range []struct {
		component, variant string
		status             int
	}{
		{"flash-next", "huihui_lil", 409},
		{"flash-next", "official", 409},
		{"flash-next", "abliterated", 409},
		{"flash-next-tp2", "huihui_lil", 400},
		{"flash-next", "invalid", 400},
	} {
		w := httptest.NewRecorder()
		body := `{"component":"` + tc.component + `","variant":"` + tc.variant + `","action":"model"}`
		s.modelPreparation(w, httptest.NewRequest("POST", "/api/models/prepare", strings.NewReader(body)))
		if w.Code != tc.status {
			t.Fatalf("%s/%s: %d %s", tc.component, tc.variant, w.Code, w.Body.String())
		}
		if tc.status == 409 && !strings.Contains(w.Body.String(), "already running") {
			t.Fatalf("request did not reach preparation lock: %s", w.Body.String())
		}
	}
}

func TestHFTokenStoredPrivatelyAndNeverReturned(t *testing.T) {
	s := &Server{cfg: config.Config{Runtime: config.RuntimeConfig{DataDir: t.TempDir()}}}
	token := "hf_unit_test_not_a_real_credential"
	w := httptest.NewRecorder()
	s.huggingFaceToken(w, httptest.NewRequest("PUT", "/api/credentials/huggingface", strings.NewReader(`{"token":"`+token+`"}`)))
	if w.Code != 200 || strings.Contains(w.Body.String(), token) {
		t.Fatal(w.Code, w.Body.String())
	}
	info, err := os.Stat(s.hfTokenPath())
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatal("token permissions", err)
	}
	w = httptest.NewRecorder()
	s.huggingFaceToken(w, httptest.NewRequest("GET", "/api/credentials/huggingface", nil))
	if strings.Contains(w.Body.String(), token) || !strings.Contains(w.Body.String(), "true") {
		t.Fatal(w.Body.String())
	}
	w = httptest.NewRecorder()
	s.huggingFaceToken(w, httptest.NewRequest("DELETE", "/api/credentials/huggingface", nil))
	if _, err = os.Stat(s.hfTokenPath()); !os.IsNotExist(err) {
		t.Fatal("token was not deleted")
	}
}
