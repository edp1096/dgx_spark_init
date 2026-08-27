package server

import (
	"archive/zip"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type preparedManifest struct {
	SourceName string            `json:"source_name"`
	Segments   []preparedSegment `json:"segments"`
	Asset      *preparedAsset    `json:"asset,omitempty"`
}

type preparedAsset struct {
	ID          string  `json:"id"`
	Filename    string  `json:"filename"`
	MediaType   string  `json:"media_type"`
	ContentType string  `json:"content_type"`
	Size        int64   `json:"size"`
	Duration    float64 `json:"duration"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
}

type preparedSegment struct {
	Name     string  `json:"name"`
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Duration float64 `json:"duration"`
}

type mediaProgressStatus struct {
	Stage           string  `json:"stage"`
	DownloadedBytes int64   `json:"downloaded_bytes"`
	TotalBytes      int64   `json:"total_bytes"`
	Percent         float64 `json:"percent"`
	ETASeconds      int     `json:"eta_seconds"`
}

func (s *Server) mediaOptions(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}
	sourceURL := strings.TrimSpace(r.FormValue("url"))
	if sourceURL == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/options"
	data, contentType, err := s.callMultipart(endpoint, map[string]string{"url": sourceURL}, "", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", valueOr(contentType, "application/json"))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *Server) prepareMediaWithProgress(j *jobs.Job, endpoint string, fields map[string]string, paths []string, archivePath string) error {
	done := make(chan error, 1)
	go func() {
		done <- s.callMultipartToFileStreaming(endpoint, fields, "file", paths, archivePath)
	}()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	defer s.clearMediaProgress(j.ID)
	last := mediaProgressStatus{}
	for {
		select {
		case err := <-done:
			if s.jobCancelled(j.ID) {
				return errJobCancelled
			}
			return err
		case <-ticker.C:
			if s.jobCancelled(j.ID) {
				return errJobCancelled
			}
			progress, err := s.getMediaProgress(j.ID)
			if err != nil || progress == last {
				continue
			}
			last = progress
			j.Params["stage"] = "media"
			j.Params["media_stage"] = progress.Stage
			if progress.TotalBytes > 0 {
				j.Params["media_percent"] = progress.Percent
				j.Params["media_downloaded_bytes"] = progress.DownloadedBytes
				j.Params["media_total_bytes"] = progress.TotalBytes
			} else {
				delete(j.Params, "media_percent")
				delete(j.Params, "media_downloaded_bytes")
				delete(j.Params, "media_total_bytes")
			}
			if progress.ETASeconds > 0 {
				j.Params["media_eta_seconds"] = progress.ETASeconds
			} else {
				delete(j.Params, "media_eta_seconds")
			}
			_ = s.jobs.Save(*j)
		}
	}
}

func (s *Server) getMediaProgress(id string) (mediaProgressStatus, error) {
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/progress/" + id
	response, err := s.health.Get(endpoint)
	if err != nil {
		return mediaProgressStatus{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return mediaProgressStatus{}, fmt.Errorf("media progress returned %d", response.StatusCode)
	}
	var progress mediaProgressStatus
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&progress); err != nil {
		return mediaProgressStatus{}, err
	}
	return progress, nil
}

func (s *Server) clearMediaProgress(id string) {
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/progress/" + id
	request, err := http.NewRequest(http.MethodDelete, endpoint, nil)
	if err != nil {
		return
	}
	response, err := s.health.Do(request)
	if err == nil {
		_ = response.Body.Close()
	}
}

func (s *Server) callMultipartToFileStreaming(url string, fields map[string]string, fileField string, paths []string, output string) error {
	return s.callMultipartToFileStreamingContext(context.Background(), url, fields, fileField, paths, output)
}

func (s *Server) callMultipartToFileStreamingContext(ctx context.Context, url string, fields map[string]string, fileField string, paths []string, output string) error {
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	producerDone := make(chan error, 1)
	go func() {
		var produceErr error
		defer func() {
			if produceErr == nil {
				produceErr = multipartWriter.Close()
			}
			_ = writer.CloseWithError(produceErr)
			producerDone <- produceErr
		}()
		keys := make([]string, 0, len(fields))
		for key := range fields {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if produceErr = multipartWriter.WriteField(key, fields[key]); produceErr != nil {
				return
			}
		}
		for _, path := range paths {
			file, err := os.Open(path)
			if err != nil {
				produceErr = err
				return
			}
			part, err := multipartWriter.CreateFormFile(fileField, filepath.Base(path))
			if err == nil {
				_, err = io.Copy(part, file)
			}
			_ = file.Close()
			if err != nil {
				produceErr = err
				return
			}
		}
	}()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, reader)
	if err != nil {
		_ = reader.CloseWithError(err)
		return err
	}
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		_ = reader.CloseWithError(err)
		<-producerDone
		return err
	}
	defer resp.Body.Close()
	producerErr := <-producerDone
	if producerErr != nil {
		return producerErr
	}
	if resp.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		return fmt.Errorf("engine returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	destination, err := os.Create(output)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(destination, resp.Body)
	closeErr := destination.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func extractPreparedArchive(archivePath, destination string) error {
	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open prepared media: %w", err)
	}
	defer archive.Close()
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return err
	}
	for _, entry := range archive.File {
		name := filepath.Clean(entry.Name)
		if filepath.Base(name) != name {
			return fmt.Errorf("invalid prepared media entry: %s", entry.Name)
		}
		source, err := entry.Open()
		if err != nil {
			return err
		}
		target, err := os.Create(filepath.Join(destination, name))
		if err == nil {
			_, err = io.Copy(target, source)
			_ = target.Close()
		}
		_ = source.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func readPreparedManifest(directory string) (preparedManifest, error) {
	file, err := os.Open(filepath.Join(directory, "manifest.json"))
	if err != nil {
		return preparedManifest{}, err
	}
	defer file.Close()
	var manifest preparedManifest
	decoder := json.NewDecoder(bufio.NewReader(file))
	if err := decoder.Decode(&manifest); err != nil {
		return preparedManifest{}, err
	}
	if len(manifest.Segments) == 0 {
		return preparedManifest{}, fmt.Errorf("prepared media contains no audio segments")
	}
	for _, segment := range manifest.Segments {
		if filepath.Base(segment.Name) != segment.Name {
			return preparedManifest{}, fmt.Errorf("invalid segment name")
		}
	}
	if manifest.Asset != nil {
		if len(manifest.Asset.ID) != 32 {
			return preparedManifest{}, fmt.Errorf("invalid prepared media asset id")
		}
		for _, char := range manifest.Asset.ID {
			if !strings.ContainsRune("0123456789abcdef", char) {
				return preparedManifest{}, fmt.Errorf("invalid prepared media asset id")
			}
		}
	}
	return manifest, nil
}
