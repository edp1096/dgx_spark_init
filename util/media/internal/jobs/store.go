package jobs

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrNotFound = errors.New("job not found")
	ErrActive   = errors.New("active job cannot be deleted")
)

type Job struct {
	ID           string            `json:"id"`
	Kind         string            `json:"kind"`
	Status       string            `json:"status"`
	Prompt       string            `json:"prompt"`
	Params       map[string]any    `json:"params,omitempty"`
	OutputURL    string            `json:"output_url,omitempty"`
	Outputs      map[string]string `json:"outputs,omitempty"`
	MediaAssetID string            `json:"media_asset_id,omitempty"`
	MediaURL     string            `json:"media_url,omitempty"`
	CaptionURL   string            `json:"caption_url,omitempty"`
	Error        string            `json:"error,omitempty"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
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
	if id == "" || filepath.Base(id) != id {
		return errors.New("invalid job id")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return ErrNotFound
	}
	if j.Status == "queued" || j.Status == "running" {
		return ErrActive
	}
	// Clean owned files first. If persistence of the record deletion fails, the
	// user can retry deletion and no untracked files are left behind.
	if err := os.RemoveAll(filepath.Join(s.dir, "inputs", id)); err != nil {
		return err
	}
	// A failed or cancelled writer may have created a job-owned file before it
	// could persist OutputURL/Outputs. Sweep the exact job namespace as well as
	// the registered URLs below so partial results cannot become orphans.
	entries, err := os.ReadDir(s.OutputDir())
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if name != id && !strings.HasPrefix(name, id+".") && !strings.HasPrefix(name, id+"-") {
			continue
		}
		if err := os.RemoveAll(filepath.Join(s.OutputDir(), name)); err != nil {
			return err
		}
	}
	if j.OutputURL != "" {
		name := filepath.Base(j.OutputURL)
		if name != "." && name != string(filepath.Separator) {
			if err := os.Remove(filepath.Join(s.OutputDir(), name)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	for _, outputURL := range j.Outputs {
		name := filepath.Base(outputURL)
		if name != "." && name != string(filepath.Separator) {
			if err := os.Remove(filepath.Join(s.OutputDir(), name)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	if j.CaptionURL != "" {
		name := filepath.Base(j.CaptionURL)
		if name != "." && name != string(filepath.Separator) {
			if err := os.Remove(filepath.Join(s.OutputDir(), name)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	delete(s.jobs, id)
	if err := s.writeLocked(); err != nil {
		s.jobs[id] = j
		return err
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
	sort.Slice(out, func(i, k int) bool {
		if !out[i].CreatedAt.Equal(out[k].CreatedAt) {
			return out[i].CreatedAt.After(out[k].CreatedAt)
		}
		if !out[i].UpdatedAt.Equal(out[k].UpdatedAt) {
			return out[i].UpdatedAt.After(out[k].UpdatedAt)
		}
		return out[i].ID > out[k].ID
	})
	return out
}

func (s *Store) OutputPath(name string) string {
	return filepath.Join(s.dir, "outputs", filepath.Base(name))
}
func (s *Store) OutputDir() string { return filepath.Join(s.dir, "outputs") }
