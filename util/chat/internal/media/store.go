package media

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	_ "golang.org/x/image/webp"
	"sparktalk/internal/db"
)

const (
	maxImageBytes  = 15 << 20
	maxImagePixels = 40_000_000
)

var imageIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

type Store struct{ dir string }

type Usage struct {
	Files       int   `json:"files"`
	Bytes       int64 `json:"bytes"`
	UnusedFiles int   `json:"unused_files"`
	UnusedBytes int64 `json:"unused_bytes"`
}

func New(databasePath string) (*Store, error) {
	dir := databasePath + ".media"
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Save(header *multipart.FileHeader) (db.Attachment, error) {
	file, err := header.Open()
	if err != nil {
		return db.Attachment{}, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxImageBytes+1))
	if err != nil {
		return db.Attachment{}, err
	}
	if len(data) == 0 || len(data) > maxImageBytes {
		return db.Attachment{}, fmt.Errorf("image must be between 1 byte and 15 MB")
	}
	mimeType := http.DetectContentType(data)
	if mimeType != "image/jpeg" && mimeType != "image/png" && mimeType != "image/webp" {
		return db.Attachment{}, fmt.Errorf("supported image types: PNG, JPEG, WebP")
	}
	config, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || config.Width < 1 || config.Height < 1 || config.Width*config.Height > maxImagePixels {
		return db.Attachment{}, fmt.Errorf("invalid image or image dimensions are too large")
	}
	id, err := randomID()
	if err != nil {
		return db.Attachment{}, err
	}
	if err := os.WriteFile(filepath.Join(s.dir, id), data, 0600); err != nil {
		return db.Attachment{}, err
	}
	name := strings.TrimSpace(filepath.Base(header.Filename))
	if name == "" || name == "." {
		name = "image"
	}
	return db.Attachment{ID: id, Name: name, MIME: mimeType, Size: int64(len(data)), URL: "/api/images/" + id}, nil
}

func (s *Store) Validate(items []db.Attachment) ([]db.Attachment, error) {
	if len(items) > 6 {
		return nil, fmt.Errorf("at most 6 images can be attached")
	}
	out := make([]db.Attachment, 0, len(items))
	for _, item := range items {
		data, mimeType, err := s.read(item.ID)
		if err != nil {
			return nil, err
		}
		name := strings.TrimSpace(filepath.Base(item.Name))
		if name == "" || name == "." {
			name = "image"
		}
		out = append(out, db.Attachment{ID: item.ID, Name: name, MIME: mimeType, Size: int64(len(data)), URL: "/api/images/" + item.ID})
	}
	return out, nil
}

func (s *Store) DataURL(item db.Attachment) (string, error) {
	data, mimeType, err := s.read(item.ID)
	if err != nil {
		return "", err
	}
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

func (s *Store) Serve(w http.ResponseWriter, r *http.Request, id string) {
	data, mimeType, err := s.read(id)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Cache-Control", "private, max-age=86400")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, _ = w.Write(data)
}

func (s *Store) Usage(referenced map[string]struct{}, keep map[string]struct{}) (Usage, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return Usage{}, err
	}
	var usage Usage
	for _, entry := range entries {
		if entry.IsDir() || !imageIDPattern.MatchString(entry.Name()) {
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
		if entry.IsDir() || !imageIDPattern.MatchString(id) {
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
	}
	return Usage{Files: before.UnusedFiles, Bytes: before.UnusedBytes}, nil
}

func (s *Store) read(id string) ([]byte, string, error) {
	if !imageIDPattern.MatchString(id) {
		return nil, "", fmt.Errorf("invalid image id")
	}
	data, err := os.ReadFile(filepath.Join(s.dir, id))
	if err != nil {
		return nil, "", err
	}
	mimeType := http.DetectContentType(data)
	if mimeType != "image/jpeg" && mimeType != "image/png" && mimeType != "image/webp" {
		return nil, "", fmt.Errorf("invalid stored image")
	}
	return data, mimeType, nil
}

func randomID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}
