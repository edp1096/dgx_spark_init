package media

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"sparktalk/internal/attachment"
	"sparktalk/internal/db"
)

// Default aliases for unconfigured callers and fixtures. Runtime paths read
// Store.Limits so setting changes apply to every attachment source.
var (
	MaxAttachmentBytes  = attachment.Default().MaxBytes()
	MaxImageBytes       = attachment.Default().ForType("image")
	MaxRemoteVideoBytes = attachment.Default().ForType("video")
	MaxMessageBytes     = attachment.Default().MessageBytes()
)

const (
	MaxInlineBytes = 64 << 20
	maxImagePixels = 40_000_000
)

var mediaIDPattern = regexp.MustCompile(`^[a-f0-9]{32}$`)

type Store struct {
	dir    string
	limits atomic.Value
}

func (s *Store) Configure(l attachment.Limits) { s.limits.Store(l.Normalized()) }
func (s *Store) Limits() attachment.Limits {
	if l := s.limits.Load(); l != nil {
		return l.(attachment.Limits)
	}
	return (attachment.Limits{}).Normalized()
}

func New(databasePath string, configured ...attachment.Limits) (*Store, error) {
	dir := databasePath + ".media"
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	s := &Store{dir: dir}
	l := attachment.Limits{}
	if len(configured) > 0 {
		l = configured[0]
	}
	s.Configure(l)
	return s, nil
}

func (s *Store) SaveImage(header *multipart.FileHeader) (db.Attachment, error) {
	return s.save(header, s.Limits().ForType("image"), true)
}

func (s *Store) SaveAttachment(header *multipart.FileHeader) (db.Attachment, error) {
	return s.save(header, s.Limits().MaxBytes(), false)
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

func (s *Store) saveReader(reader io.Reader, originalName, hint string, limit int64, imageOnly bool) (db.Attachment, error) {
	policy := s.Limits()
	limit = min(limit, policy.MaxBytes())
	if imageOnly {
		limit = min(limit, policy.ForType("image"))
	}
	if limit < 1 {
		return db.Attachment{}, fmt.Errorf("invalid attachment limit")
	}
	f, err := os.CreateTemp(s.dir, ".import-")
	if err != nil {
		return db.Attachment{}, err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	size, err := io.Copy(f, io.LimitReader(reader, limit+1))
	if err != nil {
		return db.Attachment{}, err
	}
	if size < 1 || size > limit {
		return db.Attachment{}, fmt.Errorf("file must be between 1 byte and %d MiB", limit>>20)
	}
	name := cleanName(originalName)
	kind, err := classifyFile(f, name, hint, imageOnly)
	if err != nil {
		return db.Attachment{}, err
	}
	if size > policy.ForMIME(kind) {
		return db.Attachment{}, fmt.Errorf("%s attachment exceeds %d MiB", attachment.Kind(kind), policy.ForMIME(kind)>>20)
	}
	if err = f.Close(); err != nil {
		return db.Attachment{}, err
	}
	id, err := randomID()
	if err != nil {
		return db.Attachment{}, err
	}
	if err = os.Rename(f.Name(), filepath.Join(s.dir, id)); err != nil {
		return db.Attachment{}, err
	}
	return db.Attachment{ID: id, Name: name, MIME: kind, Size: size, URL: mediaURL(id, name, kind)}, nil
}

func (s *Store) Validate(items []db.Attachment) ([]db.Attachment, error) {
	policy := s.Limits()
	if len(items) > policy.MaxFiles {
		return nil, fmt.Errorf("at most %d media files can be attached", policy.MaxFiles)
	}
	out := make([]db.Attachment, 0, len(items))
	var total int64
	for _, item := range items {
		name := cleanName(item.Name)
		file, mimeType, size, err := s.inspect(item.ID, name, item.MIME)
		if err != nil {
			return nil, err
		}
		file.Close()
		if size > policy.ForMIME(mimeType) {
			return nil, fmt.Errorf("%s exceeds %d MiB attachment limit", name, policy.ForMIME(mimeType)>>20)
		}
		total += size
		if total > policy.MessageBytes() {
			return nil, fmt.Errorf("attachments may total at most %d MB per message", policy.MessageBytes()>>20)
		}
		out = append(out, db.Attachment{ID: item.ID, Name: name, MIME: mimeType, Size: size, URL: mediaURL(item.ID, name, mimeType)})
	}
	return out, nil
}

func (s *Store) DataURL(item db.Attachment) (string, error) {
	f, kind, size, err := s.inspect(item.ID, cleanName(item.Name), item.MIME)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if strings.HasPrefix(kind, "image/") && size > MaxInlineBytes {
		data, err := inlineImage(f)
		if err != nil {
			return "", err
		}
		return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data), nil
	}

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
	file, mimeType, _, err := s.inspect(id, name, mimeHint)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Cache-Control", "private, max-age=86400")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// Source/HTML and archive links must download rather than execute as a page
	// on the Talk origin. Keep PDF and media inline for existing preview cards.
	if !strings.HasPrefix(mimeType, "image/") && !strings.HasPrefix(mimeType, "audio/") && !strings.HasPrefix(mimeType, "video/") && mimeType != "application/pdf" {
		w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": cleanName(name)}))
	}

	http.ServeContent(w, r, cleanName(name), fileModTime(filepath.Join(s.dir, id)), file)
}

func (s *Store) read(id, name, mimeHint string, imageOnly bool) ([]byte, string, error) {
	if !mediaIDPattern.MatchString(id) {
		return nil, "", fmt.Errorf("invalid media id")
	}
	info, err := os.Stat(filepath.Join(s.dir, id))
	if err != nil {
		return nil, "", err
	}
	if info.Size() > MaxInlineBytes {
		return nil, "", fmt.Errorf("attachment is too large for inline model input; use text extraction or media processing")
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

// Existing stored files remain downloadable after settings are lowered.
func (s *Store) inspect(id, name, hint string) (*os.File, string, int64, error) {
	f, err := s.Open(db.Attachment{ID: id})
	if err != nil {
		return nil, "", 0, err
	}
	fail := func(err error) (*os.File, string, int64, error) { f.Close(); return nil, "", 0, err }
	info, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	kind, err := classifyFile(f, cleanName(name), hint, false)
	if err != nil {
		return fail(err)
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return fail(err)
	}
	return f, kind, info.Size(), nil
}
