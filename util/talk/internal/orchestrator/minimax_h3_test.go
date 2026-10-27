package orchestrator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMiniMaxHealthRejectsUnrelatedService(t *testing.T) {
	correct := false
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if correct {
			w.Write([]byte(`{"status":"ok","video_model":"minimax-h3-nvfp4","features":{"progress":true,"eta":true},"video":{"steps":4},"acceleration":{"pdmd":{"modules":208}}}`))
		} else {
			w.Write([]byte(`{"status":"ok","version":"0.1.0"}`))
		}
	}))
	defer backend.Close()
	c := &Controller{client: backend.Client()}
	x := Component{ComposeAsset: "compose.qwim-mmh3.yaml", HealthURL: backend.URL}
	if c.httpHealthy(context.Background(), x) {
		t.Fatal("unrelated service accepted as MiniMax H3")
	}
	correct = true
	if !c.httpHealthy(context.Background(), x) {
		t.Fatal("qualified service rejected")
	}
}

func TestMiniMaxHealthRejectsOldOrIncompletePDMD(t *testing.T) {
	for _, payload := range []string{
		`{"status":"ok","video_model":"minimax-h3-nvfp4","features":{"progress":true,"eta":true}}`,
		`{"status":"ok","video_model":"minimax-h3-nvfp4","features":{"progress":true,"eta":true},"video":{"steps":4},"acceleration":{"pdmd":{"modules":207}}}`,
	} {
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(payload))
		}))
		c := &Controller{client: backend.Client()}
		if c.httpHealthy(context.Background(), Component{ComposeAsset: "compose.qwim-mmh3.yaml", HealthURL: backend.URL}) {
			t.Fatal("stale or incomplete PDMD runtime accepted")
		}
		backend.Close()
	}
}
