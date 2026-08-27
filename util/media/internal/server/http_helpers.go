package server

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	_ "image/jpeg"
	"io/fs"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode"
)

func newID() string { b := make([]byte, 12); _, _ = rand.Read(b); return hex.EncodeToString(b) }

func formInt(r *http.Request, k string, d int) int {
	v, e := strconv.Atoi(r.FormValue(k))
	if e != nil {
		return d
	}
	return v
}

func formInt64(r *http.Request, k string, d int64) int64 {
	v, e := strconv.ParseInt(r.FormValue(k), 10, 64)
	if e != nil {
		return d
	}
	return v
}

func formFloat64(r *http.Request, k string, d float64) float64 {
	v, e := strconv.ParseFloat(r.FormValue(k), 64)
	if e != nil {
		return d
	}
	return v
}

func valueOr(v, d string) string {
	if strings.TrimSpace(v) == "" {
		return d
	}
	return v
}

func valueIfDifferent(value, original string) string {
	if value == original {
		return ""
	}
	return value
}

func cleanEnhancedPrompt(value string) string {
	value = strings.NewReplacer(
		"\u2018", "'", "\u2019", "'", "\u201c", "\"", "\u201d", "\"",
		"\u2014", "--", "\u2013", "-", "\u00a0", " ", "\u2212", "-",
	).Replace(strings.TrimSpace(value))
	for index, char := range value {
		if unicode.IsLetter(char) {
			return strings.TrimSpace(value[index:])
		}
	}
	return ""
}

func decodeImage(data []byte) ([]byte, error) {
	var response struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("decode image response: %w", err)
	}
	if len(response.Data) == 0 || response.Data[0].B64JSON == "" {
		return nil, fmt.Errorf("image engine returned no image")
	}
	decoded, err := base64.StdEncoding.DecodeString(response.Data[0].B64JSON)
	if err != nil {
		return nil, fmt.Errorf("decode generated image: %w", err)
	}
	return decoded, nil
}

func decodeImageSeed(data []byte) (int64, bool) {
	var response struct {
		Seed *int64 `json:"seed"`
	}
	if err := json.Unmarshal(data, &response); err != nil || response.Seed == nil {
		return 0, false
	}
	return *response.Seed, true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func withLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func spaHandler(root fs.FS) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(filepath.Clean(r.URL.Path), "/")
		if p == "." {
			p = "index.html"
		}
		if _, e := fs.Stat(root, p); e != nil {
			p = "index.html"
		}
		http.ServeFileFS(w, r, root, p)
	})
}
