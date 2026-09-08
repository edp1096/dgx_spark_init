package media

import (
	"os"
	"path/filepath"
	"strings"
)

type Usage struct {
	Files       int   `json:"files"`
	Bytes       int64 `json:"bytes"`
	UnusedFiles int   `json:"unused_files"`
	UnusedBytes int64 `json:"unused_bytes"`
}

func (s *Store) Usage(referenced map[string]struct{}, keep map[string]struct{}) (Usage, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return Usage{}, err
	}
	var usage Usage
	for _, entry := range entries {
		if entry.IsDir() || !mediaIDPattern.MatchString(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return Usage{}, err
		}
		usage.Files++
		usage.Bytes += info.Size()
		if _, used := referenced[entry.Name()]; used {
			continue
		}
		if _, protected := keep[entry.Name()]; protected {
			continue
		}
		usage.UnusedFiles++
		usage.UnusedBytes += info.Size()
	}
	return usage, nil
}

func (s *Store) Cleanup(referenced map[string]struct{}, keep map[string]struct{}) (Usage, error) {
	before, err := s.Usage(referenced, keep)
	if err != nil {
		return Usage{}, err
	}
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return Usage{}, err
	}
	for _, entry := range entries {
		id := entry.Name()
		cacheSuffix := ""
		for _, suffix := range []string{".asr.json", ".document.json"} {
			if strings.HasSuffix(id, suffix) {
				cacheSuffix = suffix
				break
			}
		}
		if cacheSuffix != "" {
			mediaID := strings.TrimSuffix(id, cacheSuffix)
			if mediaIDPattern.MatchString(mediaID) {
				if _, statErr := os.Stat(filepath.Join(s.dir, mediaID)); os.IsNotExist(statErr) {
					_ = os.Remove(filepath.Join(s.dir, id))
				}
			}
			continue
		}
		if entry.IsDir() || !mediaIDPattern.MatchString(id) {
			continue
		}
		if _, used := referenced[id]; used {
			continue
		}
		if _, protected := keep[id]; protected {
			continue
		}
		if err := os.Remove(filepath.Join(s.dir, id)); err != nil && !os.IsNotExist(err) {
			return Usage{}, err
		}
		if err := os.Remove(s.transcriptPath(id)); err != nil && !os.IsNotExist(err) {
			return Usage{}, err
		}
		if err := os.Remove(s.documentPath(id)); err != nil && !os.IsNotExist(err) {
			return Usage{}, err
		}
	}
	return Usage{Files: before.UnusedFiles, Bytes: before.UnusedBytes}, nil
}
