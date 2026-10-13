package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"
)

func TestDocumentPerformanceStreamsAndPersistsFinalServerMeasurements(t *testing.T) {
	finish := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(finish) }) }
	defer release()
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 2; i++ {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking about the translation\"}}]}\n\n")
			w.(http.Flusher).Flush()
			if i == 0 {
				time.Sleep(2100 * time.Millisecond)
			}
		}
		select {
		case <-finish:
		case <-r.Context().Done():
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[],\"timings\":{\"prompt_n\":500,\"prompt_ms\":1000,\"predicted_n\":200,\"predicted_ms\":4000}}\n\n")
		r.Body = io.NopCloser(bytes.NewReader(body))
		mockDocumentResponse(w, r, true)
	})
	source := fixtureSource(t, s, "A complete paragraph to translate.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		run, err = s.db.DocumentRun("s", run.ID)
		if err != nil {
			t.Fatal(err)
		}
		if p := run.Document.Performance; p != nil && p.TG != nil {
			if !p.Live || !p.TGEstimated || p.PP != nil || p.Calls != 1 {
				t.Fatalf("live estimates must not claim exact prefill: %+v", p)
			}
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if run.Document.Performance == nil || run.Document.Performance.TG == nil {
		t.Fatal("live performance did not reach saved progress")
	}
	release()
	run = waitDocument(t, s, run.ID)
	p := run.Document.LastPerformance
	if run.Status != "completed" || run.Document.Performance != nil || p == nil || p.PP == nil || p.TG == nil || *p.PP != 500 || *p.TG != 50 || p.Live || p.PPEstimated || p.TGEstimated || p.OutputTokens != 200 {
		t.Fatalf("final measured performance not saved: status=%s metrics=%+v", run.Status, p)
	}
	units, err := s.db.DocumentUnits("s", run.ID)
	if err != nil || len(units) != 1 || len(units[0].Attempts) != 1 || units[0].Attempts[0].Performance == nil || *units[0].Attempts[0].Performance.TG != 50 {
		t.Fatal("per-call metrics not retained with checkpoint", err)
	}
}
