package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"
)

const remoteImageLimit = 32 << 20

type remoteImageRequest struct {
	URL string `json:"url"`
}

func (s *Server) fetchRemoteImage(w http.ResponseWriter, r *http.Request) {
	var input remoteImageRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 16<<10)).Decode(&input); err != nil {
		http.Error(w, "invalid image URL request", http.StatusBadRequest)
		return
	}
	parsed, err := url.Parse(strings.TrimSpace(input.URL))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		http.Error(w, "http 또는 https 이미지 URL을 입력하세요", http.StatusBadRequest)
		return
	}
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, parsed.String(), nil)
	if err != nil {
		http.Error(w, "이미지 URL을 열 수 없습니다", http.StatusBadRequest)
		return
	}
	request.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/140 Safari/537.36 SparkMedia/1.0")
	// Prefer source PNG/JPEG over CDN-generated AVIF/WebP. These images may be
	// used as diffusion conditioning where lossy conversion changes the edit.
	request.Header.Set("Accept", "image/png,image/apng,image/jpeg,image/gif,image/webp;q=0.5,*/*;q=0.1")
	client := &http.Client{
		Timeout: 30 * time.Second,
		CheckRedirect: func(next *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			if next.URL.Scheme != "http" && next.URL.Scheme != "https" {
				return fmt.Errorf("unsupported redirect scheme")
			}
			next.Header.Set("User-Agent", request.Header.Get("User-Agent"))
			next.Header.Set("Accept", request.Header.Get("Accept"))
			return nil
		},
	}
	response, err := client.Do(request)
	if err != nil {
		http.Error(w, "이미지를 내려받지 못했습니다: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		http.Error(w, fmt.Sprintf("이미지 서버가 HTTP %d을 반환했습니다", response.StatusCode), http.StatusBadGateway)
		return
	}
	if response.ContentLength > remoteImageLimit {
		http.Error(w, "이미지는 32MB 이하여야 합니다", http.StatusRequestEntityTooLarge)
		return
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, remoteImageLimit+1))
	if err != nil {
		http.Error(w, "이미지를 읽지 못했습니다", http.StatusBadGateway)
		return
	}
	if len(data) == 0 || len(data) > remoteImageLimit {
		http.Error(w, "이미지는 32MB 이하여야 합니다", http.StatusRequestEntityTooLarge)
		return
	}
	contentType, extension := remoteImageType(data, response.Header.Get("Content-Type"))
	if contentType == "" {
		http.Error(w, "지원하는 이미지 형식은 PNG, JPEG, WebP, GIF입니다", http.StatusUnsupportedMediaType)
		return
	}
	name := strings.TrimSuffix(filepath.Base(parsed.Path), filepath.Ext(parsed.Path))
	if name == "" || name == "." || name == "/" {
		name = "url-image"
	}
	name = safeRemoteImageName(name) + extension
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Header().Set("X-Image-Filename", name)
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func remoteImageType(data []byte, header string) (string, string) {
	detected := strings.ToLower(strings.TrimSpace(strings.Split(http.DetectContentType(data), ";")[0]))
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		detected = "image/webp"
	}
	if detected == "application/octet-stream" {
		detected = strings.ToLower(strings.TrimSpace(strings.Split(header, ";")[0]))
	}
	switch detected {
	case "image/png":
		return detected, ".png"
	case "image/jpeg":
		return detected, ".jpg"
	case "image/webp":
		return detected, ".webp"
	case "image/gif":
		return detected, ".gif"
	default:
		return "", ""
	}
}

func safeRemoteImageName(value string) string {
	var output strings.Builder
	for _, character := range value {
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-' || character == '_' {
			output.WriteRune(character)
		} else if output.Len() > 0 && !strings.HasSuffix(output.String(), "-") {
			output.WriteByte('-')
		}
		if output.Len() >= 80 {
			break
		}
	}
	name := strings.Trim(output.String(), "-_")
	if name == "" {
		return "url-image"
	}
	return name
}
