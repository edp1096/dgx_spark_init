package server

import (
	"net/http/httptest"
	"os"
	"sparktalk/internal/config"
	"strings"
	"testing"
)

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
