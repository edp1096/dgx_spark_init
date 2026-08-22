package media

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"sparktalk/internal/db"
)

const (
	MaxImageBytes      = 15 << 20
	MaxAttachmentBytes = 64 << 20
	MaxMessageBytes    = 96 << 20
	maxImagePixels     = 40_000_000
	maxAttachments     = 6
)

var mediaIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

type Store struct{ dir string }

func New(databasePath string) (*Store, error) {
	dir := databasePath + ".media"
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) SaveImage(header *multipart.FileHeader) (db.Attachment, error) {
	return s.save(header, MaxImageBytes, true)
}

func (s *Store) SaveAttachment(header *multipart.FileHeader) (db.Attachment, error) {
	return s.save(header, MaxAttachmentBytes, false)
}

func (s *Store) save(header *multipart.FileHeader, limit int64, imageOnly bool) (db.Attachment, error) {
	file, err := header.Open()
	if err != nil {
		return db.Attachment{}, err
	}
	defer file.Close()
	return s.saveReader(file, header.Filename, header.Header.Get("Content-Type"), limit, imageOnly)
}

// SaveReader stores a trusted media response while enforcing the same limits
// and signature checks as a browser file upload.
func (s *Store) SaveReader(reader io.Reader, name, declaredMIME string, limit int64) (db.Attachment, error) {
	return s.saveReader(reader, name, declaredMIME, limit, false)
}

func (s *Store) saveReader(reader io.Reader, originalName, declaredMIME string, limit int64, imageOnly bool) (db.Attachment, error) {
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return db.Attachment{}, err
	}
	if len(data) == 0 || int64(len(data)) > limit {
		return db.Attachment{}, fmt.Errorf("file must be between 1 byte and %d MB", limit>>20)
	}
	name := cleanName(originalName)
	mimeType, err := classifyMedia(data, name, declaredMIME, imageOnly)
	if err != nil {
		return db.Attachment{}, err
	}
	if strings.HasPrefix(mimeType, "image/") && len(data) > MaxImageBytes {
		return db.Attachment{}, fmt.Errorf("images may be at most %d MB", MaxImageBytes>>20)
	}
	id, err := randomID()
	if err != nil {
		return db.Attachment{}, err
	}
	if err := os.WriteFile(filepath.Join(s.dir, id), data, 0600); err != nil {
		return db.Attachment{}, err
	}
	return db.Attachment{ID: id, Name: name, MIME: mimeType, Size: int64(len(data)), URL: mediaURL(id, name, mimeType)}, nil
}

func (s *Store) Validate(items []db.Attachment) ([]db.Attachment, error) {
	if len(items) > maxAttachments {
		return nil, fmt.Errorf("at most %d media files can be attached", maxAttachments)
	}
	out := make([]db.Attachment, 0, len(items))
	var total int64
	for _, item := range items {
		name := cleanName(item.Name)
		data, mimeType, err := s.read(item.ID, name, item.MIME, false)
		if err != nil {
			return nil, err
		}
		total += int64(len(data))
		if total > MaxMessageBytes {
			return nil, fmt.Errorf("attachments may total at most %d MB per message", MaxMessageBytes>>20)
		}
		out = append(out, db.Attachment{ID: item.ID, Name: name, MIME: mimeType, Size: int64(len(data)), URL: mediaURL(item.ID, name, mimeType)})
	}
	return out, nil
}

func (s *Store) DataURL(item db.Attachment) (string, error) {
	data, mimeType, err := s.read(item.ID, cleanName(item.Name), item.MIME, false)
	if err != nil {
		return "", err
	}
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

// Open returns the stored attachment without loading it into memory. Callers
// use this to stream large audio/video files to local processing services.
func (s *Store) Open(item db.Attachment) (*os.File, error) {
	if !mediaIDPattern.MatchString(item.ID) {
		return nil, fmt.Errorf("invalid media id")
	}
	file, err := os.Open(filepath.Join(s.dir, item.ID))
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil || info.Size() < 1 {
		file.Close()
		if err == nil {
			err = fmt.Errorf("empty stored media")
		}
		return nil, err
	}
	return file, nil
}

func (s *Store) Serve(w http.ResponseWriter, r *http.Request, id, name, mimeHint string) {
	data, mimeType, err := s.read(id, name, mimeHint, false)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Cache-Control", "private, max-age=86400")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	http.ServeContent(w, r, cleanName(name), fileModTime(filepath.Join(s.dir, id)), bytes.NewReader(data))
}

func (s *Store) read(id, name, mimeHint string, imageOnly bool) ([]byte, string, error) {
	if !mediaIDPattern.MatchString(id) {
		return nil, "", fmt.Errorf("invalid media id")
	}
	data, err := os.ReadFile(filepath.Join(s.dir, id))
	if err != nil {
		return nil, "", err
	}
	mimeType, err := classifyMedia(data, name, mimeHint, imageOnly)
	if err != nil {
		return nil, "", fmt.Errorf("invalid stored media: %w", err)
	}
	return data, mimeType, nil
}

func cleanName(value string) string {
	name := strings.TrimSpace(filepath.Base(value))
	if name == "" || name == "." {
		return "media"
	}
	return name
}

func mediaURL(id, name, mimeType string) string {
	if strings.HasPrefix(mimeType, "image/") {
		return "/api/images/" + id
	}
	return "/api/files/" + id + "/" + url.PathEscape(cleanName(name)) + "?type=" + url.QueryEscape(mimeType)
}

func fileModTime(path string) (value time.Time) {
	if info, err := os.Stat(path); err == nil {
		return info.ModTime()
	}
	return value
}

func randomID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}
