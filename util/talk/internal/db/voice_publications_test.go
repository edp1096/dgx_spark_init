package db

import (
	"path/filepath"
	"sync"
	"testing"
)

func TestVoicePublicationConcurrentAndRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "voices.db")
	d, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"one", "two"} {
		if _, err = d.CreateSession(id, id, "model", "low"); err != nil {
			t.Fatal(err)
		}
	}
	attachment := Attachment{ID: "wav", Name: "voice.wav", MIME: "audio/wav", Size: 100, URL: "/api/media/wav"}
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m, e := d.PublishVoice("one", "job", 1, "ready", []Attachment{attachment})
			if e != nil {
				t.Error(e)
			} else if len(m.Attachments) != 1 {
				t.Error("lost audio")
			}
		}()
	}
	wg.Wait()
	messages, err := d.Messages("one")
	if err != nil || len(messages) != 1 {
		t.Fatalf("duplicate delivery: %d %v", len(messages), err)
	}
	if _, err = d.PublishVoice("two", "job", 1, "wrong", nil); err == nil {
		t.Fatal("cross-session publication allowed")
	}
	d.Close()
	d, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	m, err := d.PublishVoice("one", "job", 1, "retry", nil)
	if err != nil || m.Content != "ready" || len(m.Attachments) != 1 {
		t.Fatalf("restart lost delivery: %+v %v", m, err)
	}
	if _, err = d.PublishVoice("one", "job", 2, "new version", nil); err != nil {
		t.Fatal(err)
	}
	messages, err = d.Messages("one")
	if err != nil || len(messages) != 2 {
		t.Fatal("version missing", err)
	}
}
