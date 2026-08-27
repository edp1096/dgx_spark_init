package server

import (
	"os"
	"path/filepath"
)

func (s *Server) savedVideoInput(id, role string) (string, error) {
	dir := filepath.Join(s.dataDir, "inputs", id, role)
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			return filepath.Join(dir, entry.Name()), nil
		}
	}
	return "", nil
}
