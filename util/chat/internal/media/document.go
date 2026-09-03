package media

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type DocumentCache struct {
	Fingerprint string `json:"fingerprint"`
	Text        string `json:"text"`
	PageCount   int    `json:"page_count"`
}

func (s *Store) LoadDocument(id, fingerprint string) (DocumentCache, bool, error) {
	if !mediaIDPattern.MatchString(id) {
		return DocumentCache{}, false, fmt.Errorf("invalid media id")
	}
	data, err := os.ReadFile(s.documentPath(id))
	if os.IsNotExist(err) {
		return DocumentCache{}, false, nil
	}
	if err != nil {
		return DocumentCache{}, false, err
	}
	var cached DocumentCache
	if err := json.Unmarshal(data, &cached); err != nil {
		return DocumentCache{}, false, err
	}
	if cached.Fingerprint != fingerprint || strings.TrimSpace(cached.Text) == "" {
		return DocumentCache{}, false, nil
	}
	return cached, true, nil
}

func (s *Store) SaveDocument(id string, cached DocumentCache) error {
	if !mediaIDPattern.MatchString(id) {
		return fmt.Errorf("invalid media id")
	}
	data, err := json.Marshal(cached)
	if err != nil {
		return err
	}
	temporary, err := os.CreateTemp(s.dir, id+".document-*")
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
	return os.Rename(temporaryName, s.documentPath(id))
}

func (s *Store) documentPath(id string) string { return filepath.Join(s.dir, id+".document.json") }
