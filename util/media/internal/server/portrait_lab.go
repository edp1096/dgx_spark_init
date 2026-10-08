package server

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	portraitLabVersion      = "1.7.1"
	portraitLabSourceURL    = "https://arca.live/b/aireal/181004198"
	portraitLabGuideURL     = "https://arca.live/b/aireal/180861615"
	portraitLabDownloadURL  = "https://huggingface.co/greedjoo/ETC/resolve/main/PORTRAIT%20LAB%20by%20YDEERG_v1.7.1.zip"
	portraitLabZipSHA256    = "9b071981cd3fc41744365d3d5c5c36082f90c5f4f1ef5807b4b210750621ef56"
	portraitLabHTMLSHA256   = "75888066e940610bc96258ecbfa1bd32b125f160c9f7231114d4dc47662335c7"
	portraitLabArchiveMax   = int64(8 << 20)
	portraitLabHTMLMax      = int64(8 << 20)
	portraitLabArchiveName  = "PORTRAIT-LAB-v1.7.1-original.zip"
	portraitLabHTMLName     = "PORTRAIT-LAB-v1.7.1-original.html"
	portraitLabMetadataName = "SOURCE.txt"
)

var portraitLabWrapper = template.Must(template.New("portrait-lab").Parse(`<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>PORTRAIT LAB {{.Version}} · Spark Media</title><style>
*{box-sizing:border-box}html,body{height:100%;margin:0;background:#0d1114;color:#d8dde0;font:12px system-ui,sans-serif;overflow:hidden}
.bar{height:42px;display:flex;align-items:center;gap:12px;padding:0 max(12px,env(safe-area-inset-right)) 0 max(12px,env(safe-area-inset-left));border-bottom:1px solid #30373d;background:#151a1e}
.bar a{color:#b9e58b;text-decoration:none}.bar strong{color:#f0f2f3}.bar span{color:#737e85;font-size:10px}.bar .source{margin-left:auto;color:#95a0a7}iframe{display:block;width:100%;height:calc(100dvh - 42px);border:0;background:#f6f4ee}
@media(max-width:560px){.bar{height:38px;gap:8px}.bar span{display:none}.bar .source{font-size:10px}iframe{height:calc(100dvh - 38px)}}
</style></head><body><nav class="bar"><a href="/" target="_top">← Spark Media</a><strong>PORTRAIT LAB</strong><span>v{{.Version}} · 원본 HTML</span><a class="source" href="{{.Source}}" target="_blank" rel="noreferrer">출처 ↗</a></nav><iframe src="/tools/portrait-lab/original.html" title="PORTRAIT LAB v{{.Version}}"></iframe></body></html>`))

func (s *Server) PreparePortraitLab(ctx context.Context) error {
	s.portraitLabMu.Lock()
	defer s.portraitLabMu.Unlock()
	dir := filepath.Join(s.dataDir, "tools", "portrait-lab", "v"+portraitLabVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	archivePath := filepath.Join(dir, portraitLabArchiveName)
	htmlPath := filepath.Join(dir, portraitLabHTMLName)
	if fileSHA256(htmlPath) == portraitLabHTMLSHA256 {
		return writePortraitLabMetadata(dir)
	}
	if fileSHA256(archivePath) != portraitLabZipSHA256 {
		_ = os.Remove(archivePath)
		if err := s.ensureDownloadedFile(ctx, portraitLabDownloadURL, archivePath, portraitLabArchiveMax); err != nil {
			return fmt.Errorf("download PORTRAIT LAB v%s: %w", portraitLabVersion, err)
		}
	}
	if hash := fileSHA256(archivePath); hash != portraitLabZipSHA256 {
		return fmt.Errorf("PORTRAIT LAB archive checksum mismatch: %s", hash)
	}
	if err := extractPortraitLabHTML(archivePath, htmlPath); err != nil {
		return err
	}
	if hash := fileSHA256(htmlPath); hash != portraitLabHTMLSHA256 {
		_ = os.Remove(htmlPath)
		return fmt.Errorf("PORTRAIT LAB HTML checksum mismatch: %s", hash)
	}
	return writePortraitLabMetadata(dir)
}

func (s *Server) servePortraitLab(w http.ResponseWriter, r *http.Request) {
	if err := s.PreparePortraitLab(r.Context()); err != nil {
		http.Error(w, "prepare PORTRAIT LAB: "+err.Error(), http.StatusBadGateway)
		return
	}
	switch r.URL.Path {
	case "/tools/portrait-lab/":
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		_ = portraitLabWrapper.Execute(w, map[string]string{"Version": portraitLabVersion, "Source": portraitLabSourceURL})
	case "/tools/portrait-lab/original.html":
		path := filepath.Join(s.dataDir, "tools", "portrait-lab", "v"+portraitLabVersion, portraitLabHTMLName)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		http.ServeFile(w, r, path)
	default:
		http.NotFound(w, r)
	}
}

func extractPortraitLabHTML(archivePath, target string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open PORTRAIT LAB archive: %w", err)
	}
	defer reader.Close()
	for _, item := range reader.File {
		if item.FileInfo().IsDir() || !strings.HasSuffix(strings.ToLower(item.Name), ".html") {
			continue
		}
		if item.UncompressedSize64 == 0 || item.UncompressedSize64 > uint64(portraitLabHTMLMax) {
			return fmt.Errorf("unexpected PORTRAIT LAB HTML size: %d", item.UncompressedSize64)
		}
		source, err := item.Open()
		if err != nil {
			return err
		}
		tmp, err := os.CreateTemp(filepath.Dir(target), ".portrait-lab-*")
		if err != nil {
			_ = source.Close()
			return err
		}
		tmpName := tmp.Name()
		n, copyErr := io.Copy(tmp, io.LimitReader(source, portraitLabHTMLMax+1))
		closeSourceErr := source.Close()
		closeTargetErr := tmp.Close()
		if copyErr != nil || closeSourceErr != nil || closeTargetErr != nil || n == 0 || n > portraitLabHTMLMax {
			_ = os.Remove(tmpName)
			if copyErr != nil {
				return copyErr
			}
			return fmt.Errorf("extract PORTRAIT LAB HTML failed")
		}
		if err := os.Rename(tmpName, target); err != nil {
			_ = os.Remove(tmpName)
			return err
		}
		return nil
	}
	return fmt.Errorf("PORTRAIT LAB archive contains no HTML file")
}

func fileSHA256(path string) string {
	file, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return ""
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func writePortraitLabMetadata(dir string) error {
	content := fmt.Sprintf("PORTRAIT LAB v%s\nUpdate: %s\nGuide: %s\nDownload: %s\nZIP SHA256: %s\nHTML SHA256: %s\n", portraitLabVersion, portraitLabSourceURL, portraitLabGuideURL, portraitLabDownloadURL, portraitLabZipSHA256, portraitLabHTMLSHA256)
	return os.WriteFile(filepath.Join(dir, portraitLabMetadataName), []byte(content), 0o644)
}
