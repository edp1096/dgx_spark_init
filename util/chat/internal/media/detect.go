package media

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"path/filepath"
	"strings"

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

func hasPrefix(data []byte, value string) bool { return bytes.HasPrefix(data, []byte(value)) }

func isRIFF(data []byte, kind string) bool {
	return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == kind
}

func isISOBaseMedia(data []byte) bool { return len(data) >= 12 && string(data[4:8]) == "ftyp" }

func isMP3Frame(data []byte) bool {
	return len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0
}
