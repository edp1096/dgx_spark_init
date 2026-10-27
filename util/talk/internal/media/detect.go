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

var textTypes = map[string]string{
	".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown",
	".html": "text/html", ".htm": "text/html", ".csv": "text/csv", ".tsv": "text/tab-separated-values",
	".json": "application/json", ".xml": "application/xml", ".yaml": "text/yaml", ".yml": "text/yaml", ".toml": "text/plain",
	".js": "text/javascript", ".jsx": "text/javascript", ".ts": "text/plain", ".tsx": "text/plain",
	".css": "text/css", ".scss": "text/plain", ".py": "text/plain", ".go": "text/plain", ".rs": "text/plain",
	".java": "text/plain", ".c": "text/plain", ".h": "text/plain", ".cpp": "text/plain", ".hpp": "text/plain",
	".cs": "text/plain", ".sh": "text/plain", ".ps1": "text/plain", ".sql": "application/sql", ".ini": "text/plain",
	".conf": "text/plain", ".log": "text/plain",
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
	if kind := classifyAudioHeader(data, ext, declared); kind != "" {
		return kind, nil
	}
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
	return "", fmt.Errorf("supported attachment types: PNG, JPEG, WebP, MP3, MP2, WAV, FLAC, OGG/Opus, AAC, M4A/ALAC, AIFF, WMA, AMR, APE, CAF, MKA, AVI, MOV, MP4, WMV, WebM, ZIP, PDF, text/markup/data files, DOCX, PPTX, XLSX, ODF, EPUB, HWPX; raw PCM needs sample format information")
}

// Sniff audio containers independently of the extension. This is admission,
// not a decoding verdict: voice preparation must still probe/decode the file.
func classifyAudioHeader(data []byte, ext, declared string) string {
	switch {
	case hasPrefix(data, "fLaC"):
		return "audio/flac"
	case isRIFF(data, "WAVE"):
		return "audio/wav"
	case hasPrefix(data, "OggS"):
		if ext == ".ogv" || strings.HasPrefix(declared, "video/") {
			return "video/ogg"
		}
		return "audio/ogg"
	case len(data) >= 12 && hasPrefix(data, "FORM") && (string(data[8:12]) == "AIFF" || string(data[8:12]) == "AIFC"):
		return "audio/aiff"
	case hasPrefix(data, "#!AMR-WB\n"):
		return "audio/amr-wb"
	case hasPrefix(data, "#!AMR\n"):
		return "audio/amr"
	case hasPrefix(data, "MAC "):
		return "audio/x-ape"
	case hasPrefix(data, "caff"):
		return "audio/x-caf"
	case len(data) >= 3 && data[0] == 0xff && data[1]&0xf6 == 0xf0:
		return "audio/aac"
	case isMP3Frame(data) || (len(data) >= 10 && hasPrefix(data, "ID3") && data[3] >= 2 && data[3] <= 4):
		return "audio/mpeg"
	case isISOBaseMedia(data) && (ext == ".m4a" || ext == ".m4b" || ext == ".alac" || (len(data) >= 12 && (string(data[8:12]) == "M4A " || string(data[8:12]) == "M4B "))):
		return "audio/mp4"
	case bytes.HasPrefix(data, []byte{0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62, 0xce, 0x6c}) && (ext == ".wma" || strings.HasPrefix(declared, "audio/")):
		return "audio/x-ms-wma"
	case bytes.HasPrefix(data, []byte{0x1a, 0x45, 0xdf, 0xa3}) && (ext == ".mka" || strings.HasPrefix(declared, "audio/")):
		return "audio/x-matroska"
	}
	return ""
}

func classifyDocument(data []byte, extension, declared string) string {
	if extension == ".hwp" && bytes.HasPrefix(data, []byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1}) && bytes.Contains(data, []byte("HWP Document File\x00")) {
		return "application/x-hwp"
	}
	if extension == ".pdf" && bytes.HasPrefix(data, []byte("%PDF-")) {
		return "application/pdf"
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
	return classifyArchive(archive, extension)
}

func classifyArchive(archive *zip.Reader, extension string) string {
	if extension == ".zip" {
		return "application/zip"
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
