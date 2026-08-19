package jobs

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

var (
	ErrNotFound = errors.New("job not found")
	ErrActive   = errors.New("active job cannot be deleted")
)

type Job struct {
	ID        string         `json:"id"`
	Kind      string         `json:"kind"`
	Status    string         `json:"status"`
	Prompt    string         `json:"prompt"`
	Params    map[string]any `json:"params,omitempty"`
	OutputURL string         `json:"output_url,omitempty"`
	Error     string         `json:"error,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

type Store struct {
	mu   sync.RWMutex
	dir  string
	jobs map[string]Job
}

func New(dir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dir, "outputs"), 0o755); err != nil {
		return nil, err
	}
	s := &Store{dir: dir, jobs: map[string]Job{}}
	b, err := os.ReadFile(filepath.Join(dir, "jobs.json"))
	if err == nil {
		_ = json.Unmarshal(b, &s.jobs)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return s, nil
}

func (s *Store) Save(j Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	j.UpdatedAt = time.Now()
	s.jobs[j.ID] = j
	return s.writeLocked()
}

func (s *Store) writeLocked() error {
	b, err := json.MarshalIndent(s.jobs, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(s.dir, "jobs.json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(s.dir, "jobs.json"))
}

func (s *Store) Delete(id string) error {
	s.mu.Lock()
	j, ok := s.jobs[id]
	if !ok {
		s.mu.Unlock()
		return ErrNotFound
	}
	if j.Status == "queued" || j.Status == "running" {
		s.mu.Unlock()
		return ErrActive
	}
	delete(s.jobs, id)
	if err := s.writeLocked(); err != nil {
		s.jobs[id] = j
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()

	if id == "" || filepath.Base(id) != id {
		return errors.New("invalid job id")
	}
	if err := os.RemoveAll(filepath.Join(s.dir, "inputs", id)); err != nil {
		return err
	}
	if j.OutputURL != "" {
		name := filepath.Base(j.OutputURL)
		if name != "." && name != string(filepath.Separator) {
			if err := os.Remove(filepath.Join(s.OutputDir(), name)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	return nil
}

func (s *Store) DeleteFinished() (int, error) {
	s.mu.RLock()
	ids := make([]string, 0, len(s.jobs))
	for id, job := range s.jobs {
		if job.Status != "queued" && job.Status != "running" {
			ids = append(ids, id)
		}
	}
	s.mu.RUnlock()

	deleted := 0
	for _, id := range ids {
		if err := s.Delete(id); err != nil {
			if errors.Is(err, ErrNotFound) {
				continue
			}
			return deleted, err
		}
		deleted++
	}
	return deleted, nil
}

func (s *Store) Get(id string) (Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	j, ok := s.jobs[id]
	return j, ok
}

func (s *Store) List() []Job {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Job, 0, len(s.jobs))
	for _, j := range s.jobs {
		out = append(out, j)
	}
	sort.Slice(out, func(i, k int) bool { return out[i].CreatedAt.After(out[k].CreatedAt) })
	return out
}

func (s *Store) OutputPath(name string) string {
	return filepath.Join(s.dir, "outputs", filepath.Base(name))
}
func (s *Store) OutputDir() string { return filepath.Join(s.dir, "outputs") }
