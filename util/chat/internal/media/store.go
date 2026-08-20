package media

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	_ "golang.org/x/image/webp"
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

type TranscriptCache struct {
	Fingerprint string `json:"fingerprint"`
	Text        string `json:"text"`
	Language    string `json:"language,omitempty"`
}

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
		if strings.HasSuffix(id, ".asr.json") {
			mediaID := strings.TrimSuffix(id, ".asr.json")
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
	}
	return Usage{Files: before.UnusedFiles, Bytes: before.UnusedBytes}, nil
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

func classifyMedia(data []byte, name, declared string, imageOnly bool) (string, error) {
	detected := strings.TrimSpace(strings.Split(http.DetectContentType(data), ";")[0])
	if detected == "image/jpeg" || detected == "image/png" || detected == "image/webp" {
		config, _, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil || config.Width < 1 || config.Height < 1 || config.Width*config.Height > maxImagePixels {
			return "", fmt.Errorf("invalid image or image dimensions are too large")
		}
		return detected, nil
	}
	if imageOnly {
		return "", fmt.Errorf("supported image types: PNG, JPEG, WebP")
	}
	ext := strings.ToLower(filepath.Ext(name))
	declared = strings.ToLower(strings.TrimSpace(strings.Split(declared, ";")[0]))
	switch ext {
	case ".mp3":
		if hasPrefix(data, "ID3") || isMP3Frame(data) {
			return "audio/mpeg", nil
		}
	case ".wav":
		if isRIFF(data, "WAVE") {
			return "audio/wav", nil
		}
	case ".ogg", ".oga", ".ogv":
		if hasPrefix(data, "OggS") {
			if ext == ".ogv" || strings.HasPrefix(declared, "video/") {
				return "video/ogg", nil
			}
			return "audio/ogg", nil
		}
	case ".avi":
		if isRIFF(data, "AVI ") {
			return "video/x-msvideo", nil
		}
	case ".mov":
		if isISOBaseMedia(data) {
			return "video/quicktime", nil
		}
	case ".mp4", ".m4v":
		if isISOBaseMedia(data) {
			return "video/mp4", nil
		}
	case ".wmv":
		if bytes.HasPrefix(data, []byte{0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62, 0xce, 0x6c}) {
			return "video/x-ms-wmv", nil
		}
	case ".webm":
		if bytes.HasPrefix(data, []byte{0x1a, 0x45, 0xdf, 0xa3}) {
			return "video/webm", nil
		}
	}
	return "", fmt.Errorf("supported media types: PNG, JPEG, WebP, MP3, WAV, OGG, AVI, MOV, MP4, WMV, WebM")
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

func hasPrefix(data []byte, value string) bool { return bytes.HasPrefix(data, []byte(value)) }
func isRIFF(data []byte, kind string) bool {
	return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == kind
}
func isISOBaseMedia(data []byte) bool { return len(data) >= 12 && string(data[4:8]) == "ftyp" }
func isMP3Frame(data []byte) bool {
	return len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0
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
