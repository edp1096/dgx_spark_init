package knowledge

import (
	"archive/zip"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const MaxSourceBytes int64 = 256 << 20

type Store struct{ dir string }

type Source struct {
	Name        string
	MIMEType    string
	SizeBytes   int64
	SHA256      string
	StoragePath string
}

func New(databasePath string) (*Store, error) {
	dir := databasePath + ".knowledge"
	if err := os.MkdirAll(filepath.Join(dir, "objects"), 0700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Save(header *multipart.FileHeader) (Source, error) {
	input, err := header.Open()
	if err != nil {
		return Source{}, err
	}
	defer input.Close()
	return s.SaveReader(input, header.Filename, header.Header.Get("Content-Type"))
}

// SaveReader stores an acquired source through the same content-addressed path
// as browser uploads. The caller may stream data from a collector bundle
// without first materializing a second copy on disk.
func (s *Store) SaveReader(input io.Reader, name, declaredMIME string) (Source, error) {
	temporary, err := os.CreateTemp(s.dir, ".upload-")
	if err != nil {
		return Source{}, err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)

	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(temporary, hash), io.LimitReader(input, MaxSourceBytes+1))
	closeErr := temporary.Close()
	if copyErr != nil {
		return Source{}, copyErr
	}
	if closeErr != nil {
		return Source{}, closeErr
	}
	if written < 1 || written > MaxSourceBytes {
		return Source{}, fmt.Errorf("knowledge source must be between 1 byte and %d MB", MaxSourceBytes>>20)
	}

	name = cleanName(name)
	mimeType, err := detectSourceType(temporaryPath, name, declaredMIME)
	if err != nil {
		return Source{}, err
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	relativePath := filepath.Join("objects", digest)
	target := filepath.Join(s.dir, relativePath)
	if _, err := os.Stat(target); os.IsNotExist(err) {
		if err := os.Rename(temporaryPath, target); err != nil {
			return Source{}, err
		}
		if err := os.Chmod(target, 0600); err != nil {
			return Source{}, err
		}
	} else if err != nil {
		return Source{}, err
	}
	return Source{Name: name, MIMEType: mimeType, SizeBytes: written, SHA256: digest, StoragePath: relativePath}, nil
}

func (s *Store) Open(storagePath string) (*os.File, error) {
	path, err := s.resolve(storagePath)
	if err != nil {
		return nil, err
	}
	return os.Open(path)
}

func (s *Store) Path(storagePath string) (string, error) { return s.resolve(storagePath) }

func (s *Store) Remove(storagePath string) error {
	path, err := s.resolve(storagePath)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (s *Store) resolve(storagePath string) (string, error) {
	clean := filepath.Clean(strings.TrimSpace(storagePath))
	if clean == "." || filepath.IsAbs(clean) || clean == "objects" || !strings.HasPrefix(clean, "objects"+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid knowledge storage path")
	}
	return filepath.Join(s.dir, clean), nil
}

func NewDocumentID() (string, error) {
	data := make([]byte, 16)
	if _, err := rand.Read(data); err != nil {
		return "", err
	}
	return hex.EncodeToString(data), nil
}

func TitleFromName(name string) string {
	base := strings.TrimSpace(strings.TrimSuffix(cleanName(name), filepath.Ext(cleanName(name))))
	if base == "" {
		return "지식 문서"
	}
	return base
}

func cleanName(value string) string {
	name := strings.TrimSpace(filepath.Base(value))
	if name == "" || name == "." {
		return "knowledge"
	}
	return name
}

func detectSourceType(path, name, declared string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	header := make([]byte, 512)
	read, err := file.Read(header)
	if err != nil && err != io.EOF {
		return "", err
	}
	detected := strings.ToLower(strings.TrimSpace(strings.Split(http.DetectContentType(header[:read]), ";")[0]))
	extension := strings.ToLower(filepath.Ext(name))
	if detected == "application/pdf" || extension == ".pdf" && strings.HasPrefix(string(header[:read]), "%PDF-") {
		return "application/pdf", nil
	}
	imageTypes := map[string]bool{"image/png": true, "image/jpeg": true, "image/webp": true}
	if imageTypes[detected] {
		return detected, nil
	}
	if strings.HasPrefix(detected, "application/zip") || detected == "application/octet-stream" {
		if mimeType := detectArchiveSourceType(path, extension); mimeType != "" {
			return mimeType, nil
		}
	}
	byExtension := map[string]string{
		".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown",
		".csv": "text/csv", ".json": "application/json", ".html": "text/html",
		".htm": "text/html", ".xml": "application/xml", ".tsv": "text/tab-separated-values",
		".yaml": "text/yaml", ".yml": "text/yaml", ".toml": "text/plain", ".js": "text/javascript",
		".jsx": "text/javascript", ".ts": "text/plain", ".tsx": "text/plain", ".css": "text/css", ".scss": "text/plain",
		".py": "text/plain", ".go": "text/plain", ".rs": "text/plain", ".java": "text/plain", ".c": "text/plain",
		".h": "text/plain", ".cpp": "text/plain", ".hpp": "text/plain", ".cs": "text/plain", ".sh": "text/plain",
		".ps1": "text/plain", ".sql": "application/sql", ".ini": "text/plain", ".conf": "text/plain", ".log": "text/plain",
	}
	if mimeType := byExtension[extension]; mimeType != "" && (strings.HasPrefix(detected, "text/") || detected == "application/octet-stream" || detected == "application/json" || detected == "application/xml") {
		return mimeType, nil
	}
	declared = strings.ToLower(strings.TrimSpace(strings.Split(declared, ";")[0]))
	if strings.HasPrefix(detected, "text/") && (declared == "" || strings.HasPrefix(declared, "text/")) {
		return "text/plain", nil
	}
	return "", fmt.Errorf("unsupported knowledge source type: %s", detected)
}

func detectArchiveSourceType(filePath, extension string) string {
	archive, err := zip.OpenReader(filePath)
	if err != nil {
		return ""
	}
	defer archive.Close()
	if len(archive.File) > 10000 {
		return ""
	}
	entries := make(map[string]bool, len(archive.File))
	mimetype := ""
	for _, file := range archive.File {
		name := strings.TrimPrefix(strings.ReplaceAll(file.Name, "\\", "/"), "/")
		entries[name] = true
		if name == "mimetype" && file.UncompressedSize64 < 256 {
			reader, err := file.Open()
			if err == nil {
				data, _ := io.ReadAll(io.LimitReader(reader, 256))
				reader.Close()
				mimetype = strings.TrimSpace(string(data))
			}
		}
	}
	switch {
	case extension == ".docx" && entries["word/document.xml"]:
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	case extension == ".pptx" && entries["ppt/presentation.xml"]:
		return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
	case extension == ".xlsx" && entries["xl/workbook.xml"]:
		return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	case extension == ".hwpx" && entries["Contents/content.hpf"]:
		return "application/vnd.hancom.hwpx"
	case extension == ".epub" && (mimetype == "application/epub+zip" || entries["META-INF/container.xml"]):
		return "application/epub+zip"
	case extension == ".odt" && mimetype == "application/vnd.oasis.opendocument.text":
		return mimetype
	case extension == ".odp" && mimetype == "application/vnd.oasis.opendocument.presentation":
		return mimetype
	case extension == ".ods" && mimetype == "application/vnd.oasis.opendocument.spreadsheet":
		return mimetype
	default:
		return ""
	}
}
