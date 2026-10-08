package media

import (
	"archive/zip"
	"bytes"
	"fmt"
	"image"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// Inspect bounded headers/ZIP indexes or scan text in chunks; never allocate a
// complete uploaded archive, document or video merely to validate its format.
func classifyFile(f *os.File, name, hint string, imageOnly bool) (string, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	header := make([]byte, 512)
	n, err := io.ReadFull(f, header)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return "", err
	}
	header = header[:n]
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	detected := strings.Split(http.DetectContentType(header), ";")[0]
	if detected == "image/png" || detected == "image/jpeg" || detected == "image/webp" {
		cfg, _, err := image.DecodeConfig(io.LimitReader(f, 1<<20))
		if err != nil || cfg.Width < 1 || cfg.Height < 1 || int64(cfg.Width)*int64(cfg.Height) > maxImagePixels {
			return "", fmt.Errorf("invalid image or image dimensions are too large")
		}
		return detected, nil
	}
	if imageOnly {
		return "", fmt.Errorf("supported image types: PNG, JPEG, WebP")
	}
	ext := strings.ToLower(filepath.Ext(name))
	if kind := textTypes[ext]; kind != "" {
		if err := validateText(f); err != nil {
			return "", err
		}
		return kind, nil
	}
	if bytes.HasPrefix(header, []byte{'P', 'K'}) {
		stat, err := f.Stat()
		if err != nil {
			return "", err
		}
		z, err := zip.NewReader(f, stat.Size())
		if err != nil || len(z.File) > 10000 {
			return "", fmt.Errorf("invalid ZIP archive or too many entries")
		}
		if kind := classifyArchive(z, ext); kind != "" {
			return kind, nil
		}
	}
	return classifyMedia(header, name, hint, false)
}

func validateText(r io.Reader) error {
	buf := make([]byte, 64*1024+utf8.UTFMax)
	carry := 0
	for {
		n, err := r.Read(buf[carry : len(buf)-utf8.UTFMax])
		n += carry
		data := buf[:n]
		end := n
		if err == nil {
			// Keep a possibly incomplete final UTF-8 rune for the next chunk.
			start := n - 1
			for start >= 0 && n-start < utf8.UTFMax && !utf8.RuneStart(data[start]) {
				start--
			}
			if start >= 0 && !utf8.FullRune(data[start:]) {
				end = start
			}
		}
		if !utf8.Valid(data[:end]) || bytes.IndexByte(data[:end], 0) >= 0 {
			return fmt.Errorf("text attachment must contain UTF-8 without NUL bytes")
		}
		carry = copy(buf, data[end:])
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}
