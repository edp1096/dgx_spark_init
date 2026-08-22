package media

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type TranscriptCache struct {
	Fingerprint string `json:"fingerprint"`
	Text        string `json:"text"`
	Language    string `json:"language,omitempty"`
}

func (s *Store) LoadTranscript(id, fingerprint string) (TranscriptCache, bool, error) {
	if !mediaIDPattern.MatchString(id) {
		return TranscriptCache{}, false, fmt.Errorf("invalid media id")
	}
	data, err := os.ReadFile(s.transcriptPath(id))
	if os.IsNotExist(err) {
		return TranscriptCache{}, false, nil
	}
	if err != nil {
		return TranscriptCache{}, false, err
	}
	var cached TranscriptCache
	if err := json.Unmarshal(data, &cached); err != nil {
		return TranscriptCache{}, false, err
	}
	if cached.Fingerprint != fingerprint || strings.TrimSpace(cached.Text) == "" {
		return TranscriptCache{}, false, nil
	}
	return cached, true, nil
}

func (s *Store) SaveTranscript(id string, cached TranscriptCache) error {
	if !mediaIDPattern.MatchString(id) {
		return fmt.Errorf("invalid media id")
	}
	data, err := json.Marshal(cached)
	if err != nil {
		return err
	}
	temporary, err := os.CreateTemp(s.dir, id+".asr-*")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryName, s.transcriptPath(id))
}

func (s *Store) transcriptPath(id string) string {
	return filepath.Join(s.dir, id+".asr.json")
}
