package server

import (
	"bufio"
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	wildcardSourceURL = "https://huggingface.co/datasets/Crocody/mymuse/tree/main/Wildcards"
	wildcardMuseURL   = "https://huggingface.co/datasets/Crocody/mymuse/resolve/main/Wildcards/muse%28no_camera%29.txt"
	wildcardStyleURL  = "https://huggingface.co/datasets/Crocody/mymuse/resolve/main/Wildcards/Style.txt"
	wildcardMaxBytes  = int64(32 << 20)
)

type wildcardPromptResult struct {
	Prompt      string `json:"prompt"`
	Muse        string `json:"muse"`
	Style       string `json:"style"`
	MuseIndex   int    `json:"muse_index"`
	StyleIndex  int    `json:"style_index"`
	MuseCount   int    `json:"muse_count"`
	StyleCount  int    `json:"style_count"`
	MuseVariant string `json:"muse_variant"`
	Source      string `json:"source"`
}

// PreparePromptWildcards keeps the public Crocody wildcard dataset in the
// durable Spark Media data directory. Generation remains fully local after the
// first successful preparation.
func (s *Server) PreparePromptWildcards(ctx context.Context) error {
	s.wildcardMu.Lock()
	defer s.wildcardMu.Unlock()
	return s.loadPromptWildcardsLocked(ctx)
}

func (s *Server) randomPromptWildcard(w http.ResponseWriter, r *http.Request) {
	if err := s.PreparePromptWildcards(r.Context()); err != nil {
		http.Error(w, "prepare prompt wildcards: "+err.Error(), http.StatusBadGateway)
		return
	}

	s.wildcardMu.Lock()
	defer s.wildcardMu.Unlock()
	museIndex, err := secureRandomIndex(len(s.wildcardMuse))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	styleIndex, err := secureRandomIndex(len(s.wildcardStyles))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	muse := s.wildcardMuse[museIndex]
	style := s.wildcardStyles[styleIndex]
	writeJSON(w, http.StatusOK, wildcardPromptResult{
		Prompt:      strings.TrimSpace(muse + " " + style),
		Muse:        muse,
		Style:       style,
		MuseIndex:   museIndex + 1,
		StyleIndex:  styleIndex + 1,
		MuseCount:   len(s.wildcardMuse),
		StyleCount:  len(s.wildcardStyles),
		MuseVariant: "muse(no_camera).txt",
		Source:      wildcardSourceURL,
	})
}

func (s *Server) loadPromptWildcardsLocked(ctx context.Context) error {
	if len(s.wildcardMuse) > 0 && len(s.wildcardStyles) > 0 {
		return nil
	}
	dir := filepath.Join(s.dataDir, "prompt-wildcards", "crocody-mymuse")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	musePath := filepath.Join(dir, "muse_no_camera.txt")
	stylePath := filepath.Join(dir, "Style.txt")
	if err := s.ensureDownloadedFile(ctx, wildcardMuseURL, musePath, wildcardMaxBytes); err != nil {
		return fmt.Errorf("download muse(no_camera).txt: %w", err)
	}
	if err := s.ensureDownloadedFile(ctx, wildcardStyleURL, stylePath, wildcardMaxBytes); err != nil {
		return fmt.Errorf("download Style.txt: %w", err)
	}
	muse, err := readWildcardLines(musePath)
	if err != nil {
		return fmt.Errorf("read muse(no_camera).txt: %w", err)
	}
	styles, err := readWildcardLines(stylePath)
	if err != nil {
		return fmt.Errorf("read Style.txt: %w", err)
	}
	if len(muse) == 0 || len(styles) == 0 {
		return fmt.Errorf("wildcard source is empty")
	}
	s.wildcardMuse = muse
	s.wildcardStyles = styles
	return nil
}

func (s *Server) ensureDownloadedFile(ctx context.Context, sourceURL, target string, maxBytes int64) error {
	if info, err := os.Stat(target); err == nil && info.Size() > 0 {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "Spark-Media/1.0")
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("source returned HTTP %d", resp.StatusCode)
	}
	tmp, err := os.CreateTemp(filepath.Dir(target), ".wildcard-download-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	n, copyErr := io.Copy(tmp, io.LimitReader(resp.Body, maxBytes+1))
	closeErr := tmp.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if n == 0 || n > maxBytes {
		return fmt.Errorf("unexpected source size: %d bytes", n)
	}
	return os.Rename(tmpName, target)
}

func readWildcardLines(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	lines := make([]string, 0, 1024)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines, scanner.Err()
}

func secureRandomIndex(length int) (int, error) {
	if length < 1 {
		return 0, fmt.Errorf("cannot select from an empty wildcard list")
	}
	value, err := rand.Int(rand.Reader, big.NewInt(int64(length)))
	if err != nil {
		return 0, fmt.Errorf("select random wildcard: %w", err)
	}
	return int(value.Int64()), nil
}
