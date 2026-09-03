package media

import (
	"archive/zip"
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"path/filepath"
	"strings"
	"unicode/utf8"

	_ "golang.org/x/image/webp"
)

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
	if mimeType := classifyDocument(data, ext, declared); mimeType != "" {
		return mimeType, nil
	}
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
	return "", fmt.Errorf("supported attachment types: PNG, JPEG, WebP, MP3, WAV, OGG, AVI, MOV, MP4, WMV, WebM, PDF, text/markup/data files, DOCX, PPTX, XLSX, ODF, EPUB, HWPX")
}

func classifyDocument(data []byte, extension, declared string) string {
	if extension == ".pdf" && bytes.HasPrefix(data, []byte("%PDF-")) {
		return "application/pdf"
	}
	textTypes := map[string]string{
		".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown",
		".html": "text/html", ".htm": "text/html", ".csv": "text/csv", ".tsv": "text/tab-separated-values",
		".json": "application/json", ".xml": "application/xml", ".yaml": "text/yaml", ".yml": "text/yaml", ".toml": "text/plain",
		".js": "text/javascript", ".jsx": "text/javascript", ".ts": "text/plain", ".tsx": "text/plain",
		".css": "text/css", ".scss": "text/plain", ".py": "text/plain", ".go": "text/plain", ".rs": "text/plain",
		".java": "text/plain", ".c": "text/plain", ".h": "text/plain", ".cpp": "text/plain", ".hpp": "text/plain",
		".cs": "text/plain", ".sh": "text/plain", ".ps1": "text/plain", ".sql": "application/sql", ".ini": "text/plain",
		".conf": "text/plain", ".log": "text/plain",
	}
	if mimeType := textTypes[extension]; mimeType != "" && utf8.Valid(data) && !bytes.Contains(data, []byte{0}) {
		return mimeType
	}
	if len(data) < 4 || !bytes.HasPrefix(data, []byte{'P', 'K'}) {
		return ""
	}
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil || len(archive.File) > 10000 {
		return ""
	}
	entries := make(map[string]bool, len(archive.File))
	var mimetype string
	for _, file := range archive.File {
		name := strings.TrimPrefix(strings.ReplaceAll(file.Name, "\\", "/"), "/")
		entries[name] = true
		if name == "mimetype" && file.UncompressedSize64 < 256 {
			reader, err := file.Open()
			if err == nil {
				value := make([]byte, file.UncompressedSize64)
				read, _ := reader.Read(value)
				reader.Close()
				mimetype = strings.TrimSpace(string(value[:read]))
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
	}
	_ = declared
	return ""
}

func hasPrefix(data []byte, value string) bool { return bytes.HasPrefix(data, []byte(value)) }

func isRIFF(data []byte, kind string) bool {
	return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == kind
}

func isISOBaseMedia(data []byte) bool { return len(data) >= 12 && string(data[4:8]) == "ftyp" }

func isMP3Frame(data []byte) bool {
	return len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0
}
