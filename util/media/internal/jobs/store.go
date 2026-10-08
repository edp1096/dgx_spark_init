package jobs

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

var (
	ErrNotFound    = errors.New("job not found")
	ErrActive      = errors.New("active job cannot be deleted")
	ErrInvalidTags = errors.New("invalid job tags")
)

const (
	MaxTagsPerJob = 24
	MaxTagLength  = 32
)

type Tag struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
}

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
	Tags         []string          `json:"tags,omitempty"`
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
	j = cloneJob(j)
	// UpdateTags is the sole mutation path for existing tag associations.
	// Long-running workers may save an older Job value after a user has edited
	// tags, so always preserve the current association here.
	if current, ok := s.jobs[j.ID]; ok {
		j.Tags = append([]string(nil), current.Tags...)
	}
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
	return cloneJob(j), ok
}

func (s *Store) List() []Job {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Job, 0, len(s.jobs))
	for _, j := range s.jobs {
		out = append(out, cloneJob(j))
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

// UpdateTags atomically replaces one job's tags. Tag names are normalized for
// whitespace and case-insensitive de-duplication while retaining the first
// spelling already used by the catalog.
func (s *Store) UpdateTags(id string, requested []string) (Job, error) {
	if id == "" || filepath.Base(id) != id {
		return Job{}, ErrNotFound
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[id]
	if !ok {
		return Job{}, ErrNotFound
	}
	tags, err := normalizeTags(requested, s.canonicalTagsLocked())
	if err != nil {
		return Job{}, err
	}
	previous := cloneJob(job)
	job.Tags = tags
	job.UpdatedAt = time.Now()
	s.jobs[id] = job
	if err := s.writeLocked(); err != nil {
		s.jobs[id] = previous
		return Job{}, err
	}
	return cloneJob(job), nil
}

// Tags derives the catalog from live job associations. Consequently, deleting
// the final job using a tag removes that orphan tag without a second cleanup
// transaction or a duplicated catalog file.
func (s *Store) Tags() []Tag {
	s.mu.RLock()
	defer s.mu.RUnlock()
	counts := make(map[string]Tag)
	for _, job := range s.jobs {
		seen := make(map[string]struct{}, len(job.Tags))
		for _, name := range job.Tags {
			key := tagKey(name)
			if key == "" {
				continue
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			tag := counts[key]
			if tag.Name == "" {
				tag.Name = strings.Join(strings.Fields(name), " ")
			}
			tag.Count++
			counts[key] = tag
		}
	}
	out := make([]Tag, 0, len(counts))
	for _, tag := range counts {
		out = append(out, tag)
	}
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out
}

func (s *Store) canonicalTagsLocked() map[string]string {
	canonical := make(map[string]string)
	for _, job := range s.jobs {
		for _, name := range job.Tags {
			key := tagKey(name)
			if key != "" {
				if _, exists := canonical[key]; !exists {
					canonical[key] = strings.Join(strings.Fields(name), " ")
				}
			}
		}
	}
	return canonical
}

func normalizeTags(requested []string, canonical map[string]string) ([]string, error) {
	if len(requested) > MaxTagsPerJob {
		return nil, ErrInvalidTags
	}
	out := make([]string, 0, len(requested))
	seen := make(map[string]struct{}, len(requested))
	for _, raw := range requested {
		name := strings.Join(strings.Fields(raw), " ")
		if name == "" || utf8.RuneCountInString(name) > MaxTagLength || strings.ContainsRune(name, ',') {
			return nil, ErrInvalidTags
		}
		for _, r := range name {
			if unicode.IsControl(r) {
				return nil, ErrInvalidTags
			}
		}
		key := tagKey(name)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		if existing := canonical[key]; existing != "" {
			name = existing
		}
		out = append(out, name)
	}
	return out, nil
}

func tagKey(name string) string {
	return strings.ToLower(strings.Join(strings.Fields(name), " "))
}

// cloneJob keeps mutable request/result metadata owned by the store. Callers
// can safely update a fetched job while another goroutine lists or persists
// jobs without sharing maps or slices with the store's JSON writer.
func cloneJob(job Job) Job {
	if job.Tags != nil {
		job.Tags = append([]string(nil), job.Tags...)
	}
	if job.Params != nil {
		job.Params = cloneValue(reflect.ValueOf(job.Params)).Interface().(map[string]any)
	}
	if job.Outputs != nil {
		job.Outputs = cloneValue(reflect.ValueOf(job.Outputs)).Interface().(map[string]string)
	}
	return job
}

func cloneValue(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneValue(value.Elem())
		copy := reflect.New(value.Type()).Elem()
		copy.Set(cloned)
		return copy
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		copy := reflect.MakeMapWithSize(value.Type(), value.Len())
		iterator := value.MapRange()
		for iterator.Next() {
			copy.SetMapIndex(iterator.Key(), cloneValue(iterator.Value()))
		}
		return copy
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		copy := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		for index := 0; index < value.Len(); index++ {
			copy.Index(index).Set(cloneValue(value.Index(index)))
		}
		return copy
	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		copy := reflect.New(value.Type().Elem())
		copy.Elem().Set(cloneValue(value.Elem()))
		return copy
	default:
		return value
	}
}

func (s *Store) OutputPath(name string) string {
	return filepath.Join(s.dir, "outputs", filepath.Base(name))
}
func (s *Store) OutputDir() string { return filepath.Join(s.dir, "outputs") }
